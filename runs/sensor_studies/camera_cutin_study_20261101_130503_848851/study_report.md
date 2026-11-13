## First recorded interpretation

Run: `camera_cutin_study_20261101_130503_848851` (1,440 sensor trials).
The following describes this saved run, not a hardware benchmark.

For a rapid 4 m/s cut-in at 20 m forward gap with matched coverage and native timing:

| Metric | RGB | Event |
|---|---:|---:|
| Median first post-onset target detection | 80 ms | 40 ms |
| Mean age of target detection during motion | 104 ms | 27 ms |
| Median first post-onset angular-motion cue, conditional on cue | 380 ms | 270 ms |
| Cue before target-center lane entry | 24/30 | 30/30 |

Equalizing timing to 20 Hz / 80 ms removes the event timing advantage in this
case: median first target detection becomes 130 ms and motion cue 430 ms for event,
versus 80 ms and 380 ms for RGB. The event model cannot accumulate detections of the
stationary-relative target before motion; RGB can, which also affects track startup.
The project-default event range/noise settings weaken its results at 40 m compared
with matched coverage. Thus the original straight-ahead study missed a modeled
transverse-motion/timing advantage, but coverage and tracker design still matter.

**False-alert limitation:** At 20 m the RGB no-cut-in control produces at least one
motion cue in 13/30 six-second trials (43.3%); at 40 m, 8/30 (26.7%). These are
per-trial probabilities of any cue, not per-update false-positive rates. Event
controls produce zero target detections and zero cues because the current motion
rule suppresses static-relative targets by construction. This does not establish
real-world immunity to false alarms. The cue-before-entry results above are not
reliable safety-warning rates: noise alerts can contribute, and the simple tracker
needs false-alarm control before use as a policy input.

A 2 Hz consumer often reports a first available cue only at 0.5 s after onset,
masking differences of tens of milliseconds; some transient cues never coincide
with a decision boundary. Capturing fresher observations and acting sooner are
different outcomes. Real event optics, contrast, image processing, ego-motion,
and association are not validated by this diagnostic.


# Recorded cut-in study results

Synthetic, uncalibrated model; conditional delays must be read with availability and false alerts.

## Cut-ins

profile,gap_m,lateral_speed_m_s,kind,trials,median_detection_delay_s,detection_available_fraction,median_alert_delay_s,alert_available_fraction,median_2hz_alert_delay_s,decision_alert_available_fraction,alert_before_entry_fraction,pre_onset_false_alert_fraction,mean_detection_age_s,mean_trial_rate_rmse_deg_s,rate_available_fraction
project_defaults,20,1,rgb,30,0.08,1.0,1.66,0.8,2.75,0.1333,0.43333333333333335,0.0333,0.1044,4.107,0.963
project_defaults,20,1,event,30,0.04,1.0,1.33,0.7667,3.0,0.0667,0.6333333333333333,0.0,0.0288,3.1978,0.9412
project_defaults,20,2,rgb,30,0.08,1.0,0.68,1.0,1.5,0.3333,0.6666666666666666,0.0333,0.1041,4.1993,0.963
project_defaults,20,2,event,30,0.04,1.0,0.65,1.0,1.0,0.5333,0.7666666666666667,0.0,0.0288,3.2345,0.8821
project_defaults,20,4,rgb,30,0.08,1.0,0.38,1.0,0.5,0.6333,0.8,0.0333,0.1043,6.2997,0.9471
project_defaults,20,4,event,30,0.04,1.0,0.32,1.0,0.5,0.9333,0.9,0.0,0.0285,3.3977,0.763
project_defaults,40,1,rgb,30,0.08,1.0,2.53,0.4333,4.0,0.0333,0.2,0.0333,0.112,4.0251,0.78
project_defaults,40,1,event,30,0.06,1.0,1.99,0.8333,3.5,0.0333,0.43333333333333335,0.0,0.0437,4.2281,0.919
project_defaults,40,2,rgb,30,0.08,1.0,1.26,0.4,0.5,0.0333,0.16666666666666666,0.0333,0.1127,4.0736,0.7712
project_defaults,40,2,event,30,0.06,1.0,1.265,0.6667,2.0,0.0667,0.23333333333333334,0.0,0.0434,4.0606,0.8526
project_defaults,40,4,rgb,30,0.08,1.0,0.56,0.7,0.5,0.1667,0.3333333333333333,0.0333,0.113,4.7665,0.765
project_defaults,40,4,event,30,0.06,1.0,0.57,0.8,0.5,0.0667,0.26666666666666666,0.0,0.0437,4.2637,0.7091
matched_coverage_native_timing,20,1,rgb,30,0.08,1.0,1.66,0.8,2.75,0.1333,0.43333333333333335,0.0333,0.1044,4.107,0.963
matched_coverage_native_timing,20,1,event,30,0.04,1.0,1.305,0.8667,2.0,0.2333,0.6333333333333333,0.0,0.027,2.1294,0.942
matched_coverage_native_timing,20,2,rgb,30,0.08,1.0,0.68,1.0,1.5,0.3333,0.6666666666666666,0.0333,0.1041,4.1993,0.963
matched_coverage_native_timing,20,2,event,30,0.04,1.0,0.41,1.0,1.0,0.8,1.0,0.0,0.0268,2.1874,0.8838
matched_coverage_native_timing,20,4,rgb,30,0.08,1.0,0.38,1.0,0.5,0.6333,0.8,0.0333,0.1043,6.2997,0.9471
matched_coverage_native_timing,20,4,event,30,0.04,1.0,0.27,1.0,0.5,1.0,1.0,0.0,0.0267,2.3219,0.7663
matched_coverage_native_timing,40,1,rgb,30,0.08,1.0,2.53,0.4333,4.0,0.0333,0.2,0.0333,0.112,4.0251,0.78
matched_coverage_native_timing,40,1,event,30,0.04,1.0,1.625,0.2667,,0.0,0.16666666666666666,0.0,0.03,2.2229,0.9404
matched_coverage_native_timing,40,2,rgb,30,0.08,1.0,1.26,0.4,0.5,0.0333,0.16666666666666666,0.0333,0.1127,4.0736,0.7712
matched_coverage_native_timing,40,2,event,30,0.04,1.0,0.91,0.5,2.0,0.0667,0.26666666666666666,0.0,0.03,2.2779,0.8804
matched_coverage_native_timing,40,4,rgb,30,0.08,1.0,0.56,0.7,0.5,0.1667,0.3333333333333333,0.0333,0.113,4.7665,0.765
matched_coverage_native_timing,40,4,event,30,0.04,1.0,0.47,0.9667,1.0,0.5333,0.5333333333333333,0.0,0.03,2.4203,0.7596
matched_coverage_equal_timing,20,1,rgb,30,0.08,1.0,1.66,0.8,2.75,0.1333,0.43333333333333335,0.0333,0.1044,4.107,0.963
matched_coverage_equal_timing,20,1,event,30,0.13,1.0,2.06,0.7333,3.25,0.1333,0.3333333333333333,0.0,0.1044,4.0165,0.8759
matched_coverage_equal_timing,20,2,rgb,30,0.08,1.0,0.68,1.0,1.5,0.3333,0.6666666666666666,0.0333,0.1041,4.1993,0.963
matched_coverage_equal_timing,20,2,event,30,0.13,1.0,0.845,0.8667,1.5,0.2,0.5,0.0,0.1044,3.8384,0.7886
matched_coverage_equal_timing,20,4,rgb,30,0.08,1.0,0.38,1.0,0.5,0.6333,0.8,0.0333,0.1043,6.2997,0.9471
matched_coverage_equal_timing,20,4,event,30,0.13,1.0,0.43,1.0,0.5,0.6667,0.6333333333333333,0.0,0.1041,3.923,0.6145
matched_coverage_equal_timing,40,1,rgb,30,0.08,1.0,2.53,0.4333,4.0,0.0333,0.2,0.0333,0.112,4.0251,0.78
matched_coverage_equal_timing,40,1,event,30,0.13,1.0,1.91,0.3,,0.0,0.16666666666666666,0.0,0.1122,4.0356,0.7037
matched_coverage_equal_timing,40,2,rgb,30,0.08,1.0,1.26,0.4,0.5,0.0333,0.16666666666666666,0.0333,0.1127,4.0736,0.7712
matched_coverage_equal_timing,40,2,event,30,0.13,1.0,0.93,0.3667,,0.0,0.2,0.0,0.1118,3.9057,0.6407
matched_coverage_equal_timing,40,4,rgb,30,0.08,1.0,0.56,0.7,0.5,0.1667,0.3333333333333333,0.0333,0.113,4.7665,0.765
matched_coverage_equal_timing,40,4,event,30,0.13,1.0,0.66,0.5667,0.5,0.0667,0.16666666666666666,0.0,0.1105,4.1269,0.5111

## Stationary controls

profile,gap_m,kind,trials,false_alert_fraction,mean_target_updates
project_defaults,20,rgb,30,0.43333333333333335,107.0
project_defaults,20,event,30,0.0,0.0
project_defaults,40,rgb,30,0.26666666666666666,93.8333
project_defaults,40,event,30,0.0,0.0
matched_coverage_native_timing,20,rgb,30,0.43333333333333335,107.0
matched_coverage_native_timing,20,event,30,0.0,0.0
matched_coverage_native_timing,40,rgb,30,0.26666666666666666,93.8333
matched_coverage_native_timing,40,event,30,0.0,0.0
matched_coverage_equal_timing,20,rgb,30,0.43333333333333335,107.0
matched_coverage_equal_timing,20,event,30,0.0,0.0
matched_coverage_equal_timing,40,rgb,30,0.26666666666666666,93.8333
matched_coverage_equal_timing,40,event,30,0.0,0.0
