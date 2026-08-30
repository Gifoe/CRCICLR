# Leakage audit

Banks used training partitions only. Source selection used OpenBMI sessions 1/2 and WBCIC S1/S2. WBCIC S3 was opened only if source gates passed and only after a committed protocol lock. Outer and sealed resources were never opened.
