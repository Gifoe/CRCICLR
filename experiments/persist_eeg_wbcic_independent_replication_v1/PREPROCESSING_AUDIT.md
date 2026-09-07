# Preprocessing audit

The historical primary WBCIC preprocessing is unchanged: 59 EEG channels at 1000 Hz; subtract Pz and drop it (58 channels); 0.5–40 Hz zero-phase fourth-order Butterworth SOS; `resample_poly` to 250 Hz; 0–4 s relative to the BIDS MI event with no additional offset; microvolts/20 clipped to [-12.5,12.5]. No cross-subject amplitude normalization is applied.
