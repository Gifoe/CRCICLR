# Preprocessing smoke-test report

Two subjects per dataset were read, channel-selected, filtered, resampled, windowed, atomically cached, resumed, and episode-checked before the full run.

| dataset | subject | windows | checks |
| --- | --- | --- | --- |
| eegmmidb | eegmmidb:001 | 90 | quality_flags present; episode valid |
| eegmmidb | eegmmidb:002 | 90 | quality_flags present; episode valid |
| hmc | hmc:001 | 854 | quality_flags present; episode valid |
| hmc | hmc:002 | 856 | quality_flags present; episode valid |
| cap | cap:brux1 | 426 | quality_flags present; episode valid |
| cap | cap:brux2 | 1029 | quality_flags present; episode valid |

The smoke run detected the CAP clock-format inconsistency (`HH.MM.SS` versus `HH:MM:SS`); the parser and regression test now cover both formats.
