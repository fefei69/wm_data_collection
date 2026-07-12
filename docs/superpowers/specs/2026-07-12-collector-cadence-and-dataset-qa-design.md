# Collector Cadence and Dataset QA Design

## Goal

Ensure every saved keyboard-collection episode satisfies the specified 5 Hz
command cadence, and make the offline dataset checker reject invalid data with
clear failures instead of silently accepting or crashing.

## Control-loop design

Preview rendering remains capped at 30 Hz. It is scheduled independently from
the control tick and may display the same latest camera image more than once.
The main loop polls events and timing frequently without allowing preview
pacing to block the 5 Hz control deadline.

Recording continues to consume exactly one fresh, previously unused image per
5 Hz tick. The external camera's measured rate of about 58 Hz is sufficient:
roughly eleven new frames arrive between recording ticks, and the collector
selects only the newest eligible frame.

The collector enforces the same 0.20 +/- 0.02 second cadence used by offline
QA. If a scheduled tick is too late, or if two actual command timestamps fall
outside that interval, the active episode is discarded and the schedule is
realigned without issuing catch-up ticks. Idle operation may continue safely.

## Dataset-checker design

The action/proprio alignment check distinguishes an uninformative all-zero
action sequence from commanded motion with undefined correlation. If nonzero
motion was commanded but achieved displacement has zero variance, the episode
fails because the robot did not measurably follow the commands. An all-zero
episode remains outside this numeric correlation check and is left to coverage
review.

Before episode indexing, the schema checker requires `ep_len` and `ep_offset`
to be one-dimensional integer datasets of equal shape. Every episode length
must be positive, offsets must be nonnegative running sums, and their total
must match the column row count. Invalid bookkeeping produces schema failures
and skips episode checks; it never raises an indexing exception.

## Error handling

Timing violations use the collector's existing transactional discard path, so
no partial episode reaches HDF5. Schema and alignment violations are accumulated
as normal `FAIL` results so the checker exits 1 while continuing to report other
files.

## Testing

Regression tests will cover:

- independent preview/control scheduling and detection of a command interval
  outside 180-220 ms;
- undefined correlation with nonzero commanded motion;
- zero-length and malformed bookkeeping arrays returning schema failures;
- the existing camera, collector, and dataset-checker behavior.

Tests will be written and observed failing before production changes. The
hardware-independent suite will run in a temporary environment containing
NumPy, h5py, pytest, and headless OpenCV.
