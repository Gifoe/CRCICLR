def train_model(model: torch.nn.Module, name: str, bundle: Any, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], cache: Any, manifest_sha: str, cell_dir: Path, device: torch.device) -> dict[str, Any]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cell_dir / 'checkpoint_latest.pt'
    init_hash = state_hash(copy.deepcopy(model.state_dict()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0005)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    (start, history, best, best_epoch, best_state) = (1, [], -float('inf'), None, None)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['init_sha256'] != init_hash or saved['manifest_sha256'] != manifest_sha:
            raise RuntimeError(f'resume invariant mismatch for {checkpoint}')
        model.load_state_dict(saved['current_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start = int(saved['epoch']) + 1
        (history, best, best_epoch, best_state) = (saved['history'], saved['best_val_BA'], saved['best_epoch'], saved['best_state'])
        restore_rng(saved['rng'])
    started = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for episode in manifest[epoch - 1]:
            indices = episode['support_indices'] + episode['query_indices']
            (x, y) = cache.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                (logits, _) = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite CE in {name}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        (val_rows, val_ba, _) = carrier.eval_rows(model, bundle, fold['inner_val_subjects'], cache)
        selected = epoch >= MIN_EPOCH and val_ba > best + 1e-12
        if selected:
            (best, best_epoch, best_state) = (float(val_ba), epoch, copy.deepcopy(model.state_dict()))
        row = {'epoch': epoch, 'CE': float(np.mean(losses)), 'inner_val_subject_BA': float(val_ba), 'selected': bool(selected)}
        history.append(row)
        observe(epoch, optimizer, scaler)
        torch.save({'epoch': epoch, 'history': history, 'best_val_BA': best, 'best_epoch': best_epoch, 'best_state': best_state, 'current_state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(), 'manifest_sha256': manifest_sha, 'init_sha256': init_hash}, checkpoint)
        if epoch == 1 or epoch % 5 == 0 or selected:
            print(f"[{bundle.name} fold={fold['fold_id']} seed={cell_dir.name} {name}] epoch={epoch:02d} CE={row['CE']:.4f} valBA={val_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError(f'no eligible checkpoint for {name}')
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    epoch60_state = saved['current_state']
    epoch60_path = cell_dir / 'epoch60.pt'
    torch.save(epoch60_state, epoch60_path)
    model.load_state_dict(best_state)
    selected_path = cell_dir / 'selected_best.pt'
    torch.save(model.state_dict(), selected_path)
    return {'model': name, 'selected_epoch': int(best_epoch), 'best_inner_val_BA': float(best), 'epoch60_inner_val_BA': float(history[-1]['inner_val_subject_BA']), 'history': history, 'checkpoint_path': str(selected_path), 'epoch60_checkpoint_path': str(epoch60_path), 'checkpoint_sha256': sha_file(selected_path), 'epoch60_checkpoint_sha256': sha_file(epoch60_path), 'init_sha256': init_hash, 'manifest_sha256': manifest_sha, 'elapsed_seconds': time.perf_counter() - started, 'amp': amp, 'epochs_completed': len(history)}
