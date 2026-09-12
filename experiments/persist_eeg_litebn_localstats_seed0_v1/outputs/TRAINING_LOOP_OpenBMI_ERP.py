def train_one(model: torch.nn.Module, name: str, task: str, fold: dict[str, Any], bundle: Any, cache: RawGPUCache, mean: np.ndarray, std: np.ndarray, weights: torch.Tensor | None, weight_info: dict[str, Any], device: torch.device, lock_hash: str) -> dict[str, Any]:
    cell = RUNTIME / f"{task.lower()}_fold{fold['fold_id']}_seed{fold['_seed']}_{name.lower()}"
    cell.mkdir(parents=True, exist_ok=True)
    latest = cell / 'checkpoint_latest.pt'
    selected = cell / 'selected_best.pt'
    init = state_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    train_idx = bundle.indices(fold['inner_train_subjects'], (1,))
    manifest_hash = hashlib.sha256(train_idx.tobytes() + f"{task}:{fold['fold_id']}:full_permutation_batch64".encode()).hexdigest()
    (start, history, best, best_epoch, best_state) = (1, [], -float('inf'), None, None)
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        invariant = {'init_sha256': init, 'manifest_sha256': manifest_hash, 'lock_sha256': lock_hash, 'class_weight_info': weight_info}
        if any((saved[k] != v for (k, v) in invariant.items())):
            raise RuntimeError(f'resume invariant mismatch {cell}')
        model.load_state_dict(saved['current_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'])
        (start, history, best, best_epoch, best_state) = (int(saved['epoch']) + 1, saved['history'], float(saved['best']), saved['best_epoch'], saved['best_state'])
    started = time.perf_counter()
    class_weight = None if weights is None else weights.to(device)
    for epoch in range(start, EPOCHS + 1):
        model.train()
        losses = []
        for batch in epoch_batches(train_idx, int(fold['fold_id']), task, epoch):
            (x, y) = cache.batch(batch, mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                (logits, _) = model(x)
                loss = F.cross_entropy(logits, y, weight=class_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss {task}/{name}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_val(model, bundle, cache, fold['inner_val_subjects'], mean, std)
        chosen = epoch >= MIN_EPOCH and val > best + 1e-12
        if chosen:
            (best, best_epoch, best_state) = (float(val), int(epoch), copy.deepcopy(model.state_dict()))
        history.append({'epoch': int(epoch), 'weighted_CE': float(np.mean(losses)), 'inner_val_subject_BA': float(val), 'selected': bool(chosen), 'batches': len(losses)})
        observe(epoch, optimizer, scaler)
        torch.save({'epoch': epoch, 'history': history, 'best': best, 'best_epoch': best_epoch, 'best_state': best_state, 'current_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(), 'init_sha256': init, 'manifest_sha256': manifest_hash, 'lock_sha256': lock_hash, 'class_weight_info': weight_info}, latest)
    if best_state is None:
        raise RuntimeError(f'no checkpoint eligible after epoch {MIN_EPOCH}')
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), selected)
    return {'task': task, 'fold': int(fold['fold_id']), 'seed': int(fold['_seed']), 'model': name, 'checkpoint_path': str(selected), 'checkpoint_sha256': sha256(selected), 'selected_epoch': int(best_epoch), 'best_inner_val_BA': float(best), 'init_sha256': init, 'manifest_sha256': manifest_hash, 'normalizer': weight_info.pop('_normalizer'), 'class_weight_info': weight_info, 'history': history, 'elapsed_seconds': time.perf_counter() - started, 'source_train_only': True}
