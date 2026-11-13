## First recorded results

Recorded run: `sensor_contribution_study_20261101_124523_070422`.
These results describe this saved run; reruns may differ if configuration/models change.

- All-on completed 6/6 matched validation trials with zero crashes.
- Each individual modality and every leave-one-modality-out condition also completed 6/6.
  All of these produced the same full driving sequence as all-on on each matched trial.
- All-off completed 4/6 and crashed in 2/6, with different driving sequences on all six trials.
- Equal 60 m, equal 100 m, original RGB100/event60, and reversed RGB60/event100 profiles
  all completed 6/6 in 15 s on average. Map support/uncertainty changed, but aggregate driving
  outcomes did not. These proxies do not establish perception accuracy.

Interpretation: this frozen short-route driver is insensitive to many sensing changes in
these scenarios. Its response to all-off could depend on sensor-status features as well as
missing detections; this experiment does not separate those causes. No modality is proven
superior, and the optical cutoffs remain uncalibrated. Distribution shift and the six-trial
sample limit the conclusions. A useful next diagnostic would separate sensor-status inputs
from delivered measurement information and add controlled closing-vehicle/lane-change
scenarios before using driving outcomes to rank sensing configurations.
