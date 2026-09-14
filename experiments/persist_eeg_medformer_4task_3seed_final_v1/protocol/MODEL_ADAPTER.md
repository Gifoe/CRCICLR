# Model adapter

The cache boundary is `B,C,T`; official Medformer classification consumes `B,T,C`. The adapter performs only this transpose and supplies dynamic channel, sequence-length, and class-count fields. No feature engineering, filtering, resampling, augmentation, SWA, calibration, or task-specific architecture change is performed.

