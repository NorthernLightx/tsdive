# Glossary

The words tsdive prints and the guide pages use, in alphabetical order.
Numbers in the examples come from the demo archives that `tsdive demo`
writes.

## Adjusted estimate {#adjusted-estimate}

The switchback difference B - A after an ordinary least squares fit
removes the part of the target that the covariates explain. On the demo
trial, where setting B adds 0.25 degC, it is +0.2847 degC against a
direct estimate of +0.6024 degC: feed flow and ambient temperature moved
the outlet temperature too, and the fit takes that part out. Trust it
only when every covariate is a disturbance the setting cannot move.

## Aggregate type {#aggregate-type}

The third field of the sampling contract: whether the export holds raw
samples (`NONE`) or values a historian aggregated over intervals, such
as averages.

## Archive {#archive}

One parquet file holding one tag: the timestamps in UTC, the values, the
raw quality codes, and the tag's metadata under the `tsdive.meta` key.
`tsdive ingest` and `write_tag` create archives, and nothing edits one
afterwards.

## Baseline {#baseline}

A stretch of history you trust, which a later window is judged against.
A baseline needs 30 GOOD samples and no clipped sample, else tsdive
raises `InsufficientQuality`.

## Block {#block}

One stretch of a switchback trial that runs one setting, A or B, from
start to end. The demo trial has 24 blocks of 1 h.

## Calculation basis {#calculation-basis}

How a statistic weights the samples. `TIME_WEIGHTED` weights each sample
by the time it held, which suits a value that holds until the next
sample. `EVENT_WEIGHTED` weights every sample the same, which suits
counts and events. It is the first field of the sampling contract.

## Censored {#censored}

A window is censored when at least one sample sits at an end of the
engineering range or carries an Over Range or Under Range state. Such a
sample says only that the true value was at least, or at most, that end.
`censored unknown` means the tag declares no engineering range, so
tsdive cannot tell.

## Changes per hour {#changes-per-hour}

`changes/h` in the profile: how many times per hour a GOOD value differs
from the one before it. 53.20 on the demo flow; a value near 0 on a
measurement suggests a frozen sensor or heavy compression.

## Clipped {#clipped}

`clipped` in the profile: the share of samples at an end of the
engineering range, within 1e-9 times the span. 0.0516 on the demo flow: 29
of 562 samples sat at 100 m3/h. A clipped window is censored.

## Clearing {#clearing}

A pair in `compare` clears when the 95% interval on its change in
correlation excludes 0. `pairs 1 of 1 clearing` in the demo means the
one pair of tags changed by more than its resampling noise.

## Clock control {#clock-control}

A score that reads only a window's position in its record and no sensor
value. The evaluation protocol publishes it beside every detector, so a
detector's result can be read against what time alone gives.

## Contributors {#contributors}

The tags that carry the most of a T2 or SPE breach in `mspc` and
`compare`, each with its share. With three tags or fewer every tag would
be listed, so `mspc` prints `not ranked`.

## Covariate {#covariate}

A tag passed to `switchback analyze` with `--covariate` to explain part
of the target's noise, such as feed flow or ambient temperature. A
covariate the setting itself moves (OP, SP, the manipulated flow) takes
the effect away with it.

## Coverage {#coverage}

The share of a window's time that is not lost to a data-loss gap. The
demo flow loses 40 of 600 minutes, so its coverage is 0.933.

## Data-loss gap {#data-loss-gap}

A gap whose class is `unknown`, `comm_outage_multitag` or `scan_off`:
time for which the historian should hold samples and holds none. Only
data-loss gaps lower coverage.

## Decoupled pair {#decoupled-pair}

Two tags whose correlation, over first differences, dropped between the
before and after periods of `compare`. The demo flow and temperature go
from 0.69 to 0.17.

## Digest {#digest}

A short hash printed with a result. The contract digest identifies the
sampling contract, not the data, so two tags read under one contract
show the same digest. The plan digest of a switchback plan covers its
schedule, so an edited plan file no longer matches.

## Direct estimate {#direct-estimate}

The switchback difference B - A in means over the kept samples, with no
covariate. It rests on the randomization alone.

## Distinct {#distinct}

`distinct` in the profile: how many different GOOD values the window
holds. 533 of 561 on the demo flow.

## DST {#dst}

A daylight saving time change. `--tz` names the zones whose changes
inside the window the profile lists. The demo window holds the change in
Europe/London at 01:00 UTC on 31 March 2024.

## Edge slack {#edge-slack}

Time between a window's start and the first sample, or between the last
sample and the window's end, shorter than the gap threshold. It is
counted as covered.

## Empirical quantile {#empirical-quantile}

A limit read from the fitted data itself: `q0.99` is the value 99% of
the baseline rows stay under. `mspc` sets its T2 and SPE limits this
way.

## Engineering range {#engineering-range}

The span the transmitter measures, from `eng_range_zero` to
`eng_range_zero + eng_range_span`, 0 to 100 m3/h on FIC-101. At either
end the reading stops meaning what it says.

## Evidence ledger {#evidence-ledger}

What `tsdive run` writes: every profile, finding and refusal of a plan,
as `ledger.json`, `ledger.txt` and `report.html`.

## Explained variance {#explained-variance}

`explained` in `mspc` and `compare`: the share of the baseline's
variance the kept PCA components carry. For the two demo tags it is
0.9767 from 20:00 to 23:00 and 0.5129 from 04:00 to 06:00, after the
temperature stops tracking the flow.

## Finding {#finding}

One answer in an evidence ledger, the result of one step on one or more
archives.

## Flagged {#flagged}

A sample outside the screen limits. `flagged 29 of 300 (9.7%)` counts
them over the GOOD samples of the window. In `compare`, `flagged` is the
share of the after period that a screen built on the before period
flags.

## Flatline {#flatline}

A frozen sensor: a value that stops changing although the process
moves. `profile --flatline` compares the window with earlier windows of
the same length in the same archive. It reports `NOT ASSESSED` when the
window is censored, because a sensor saturated at full scale also
reads flat.

## Gap {#gap}

A hole between two samples longer than the gap threshold, or at a
window's edge. Every gap gets a class and the rule that gave it.

## Gap class {#gap-class}

The reason tsdive gives a gap, first match wins: `scan_off` inside a
declared scan-off period, `comm_outage_multitag` when a peer tag of the
same source has the same hole, `compression_steady` for a stepped tag
with a hole within 50 times its median interval, `sparse_by_design` for
a hole within 3 times the declared sample rate, and `unknown` when no
rule matches. `unknown` is a data-loss class.

## Gap threshold {#gap-threshold}

The spacing above which a hole counts as a gap: 2.5 times the median
interval, and at least 60 s. 150 s on the 60 s demo flow.

## Historian {#historian}

The plant database that stores tag values over time, such as OSIsoft PI,
AspenTech IP.21, an OPC UA server's history or a SCADA archive. tsdive
reads what a historian exported, never the historian itself.

## Identity {#identity}

A tag's coordinates, `source_id:point_id`, such as `demo:FIC101.PV`.
`source_id` names the historian or collector and `point_id` the tag in
it. The display name is metadata and can change; the identity cannot.

## Individuals chart {#individuals-chart}

A control chart of single samples, with a center line and limits 3
sigma either side of it. `tsdive spc` draws its limits from a baseline
and runs the SPC rules on the window.

## Interval {#interval}

The time between successive samples. The profile prints its median, 5th
and 95th percentiles, and the declared sample rate beside them.

## Level shift {#level-shift}

`sigma` in the `compare` table: how far the after period's median moved
from the before period's, in before-period sigmas. +0.8 for the demo
temperature.

## MAD {#mad}

Median absolute deviation: the median distance of the samples from their
median. Times 1.4826 it estimates sigma for normal data, and one wild
sample cannot move it. 0.4017 m3/h over the whole demo flow window.

## Metadata {#metadata}

The facts about a tag the samples cannot tell: identity, name, unit,
engineering range, sample rate, role, quality codes. It lives in the
archive under `tsdive.meta` and comes from a JSON file at ingest.

## MODE tag {#mode-tag}

A tag whose values are string states, such as a recipe step or a pump
running or stopped. `role: MODE` switches off numeric statistics for it.
`screen --mode` builds one baseline per state it holds.

## Naive timestamp {#naive-timestamp}

A timestamp with no zone or offset, such as `02/03/2026 08:00`. tsdive
does not assume UTC for it: `ingest --tz` states the zone it was written
in.

## p-value (randomization) {#randomization-p-value}

The share of schedules the plan could have drawn that give a difference
at least as far from 0 as the observed one, had the setting done
nothing. `p 0.154` means about 154 in 1000 reshuffled schedules gave a
gap as big as the trial's.

## PCA {#pca}

Principal component analysis: rotating several correlated tags into a
few components that carry most of their joint variance. `mspc` and
`compare` fit it on a baseline and read later rows against it.

## Plan {#plan}

For `tsdive run`, a TOML file naming archives, windows and steps. For a
switchback trial, the JSON file `switchback plan` writes: the blocks,
their settings and the digest.

## Power readout {#power-readout}

What `switchback plan --history` prints: how often the plan's design
would claim a difference when a shift of a given size, in sigma, is
added to the B blocks of a history window where nothing changed. At a
shift of 0 it is the false-claim rate, near 0.05.

## Provisional {#provisional}

The caveat on a screen with one baseline for every regime in the
window. A regime change inside the window reads as flagged samples,
which is right only if the regimes should behave alike.

## Quality code {#quality-code}

The raw status a historian stores beside each value, such as `GOOD`,
`192`, `Questionable` or `I/O Timeout`. tsdive keeps it verbatim and
derives a severity from it.

## Refusal {#refusal}

A typed error that derives from `TSDiveError`: the data cannot answer
the question, and the message names the check and the reason. A refusal
exits with status 3 and is a result in its own right.

## Regime {#regime}

A stretch where the process runs at one operating point, such as a
throughput step or a recipe. `segment` finds regimes in the samples;
a MODE tag records them.

## Retrieval mode {#retrieval-mode}

How the export got its samples: `RECORDED` values as the historian
stored them, or `INTERPOLATED` values computed at fixed times. It is
the second field of the sampling contract, stated in the metadata.

## Role {#role}

What a tag is in its control loop: `PV` the measured process value,
`SP` the setpoint, `OP` the controller output, `MODE` a string state.

## Sample rate {#sample-rate}

`sample_rate_s` in the metadata: the scan rate the historian was set to,
in seconds. `compare` and `mspc` align tags on it, and a hole of up to 3
times it counts as expected spacing.

## Sampling contract {#sampling-contract}

The four facts every read states about how its numbers were made:
calculation basis, retrieval mode, aggregate type and stepped
interpolation. `TIME_WEIGHTED  RECORDED  NONE  stepped no` is the
default. Two reads under different contracts are different quantities.

## Segment {#segment}

A stretch between two changepoints that `tsdive segment` finds, with its
median and MAD. PELT with an L2 cost on MAD-scaled values finds them.

## Severity {#severity}

GOOD, UNCERTAIN or BAD, derived from the quality code. Only GOOD samples
with a finite value feed the statistics. A code tsdive cannot read is
UNCERTAIN, never GOOD.

## Sigma {#sigma}

The spread tsdive works in, 1.4826 times the MAD of a baseline. 0.5488
m3/h for the demo baseline from 20:00 to 01:00.

## SPC rule {#spc-rule}

A test on an individuals chart. tsdive runs three: `BEYOND_3SIGMA` for a
sample outside the 3-sigma limits, `RUN_9_SAMESIDE` for 9 samples in a
row on one side of the center, `TREND_6` for 6 samples in a row rising
or falling.

## SPE {#spe}

Squared prediction error: how far a row sits from the plane the PCA
model allows, the part of the row the correlations do not explain. A
tag that stops tracking its partners shows in SPE.

## Spread ratio {#spread-ratio}

`spread` in the `compare` table: the after period's MAD over the before
period's. x6.0 for the demo temperature, which got six times noisier.

## Stall {#stall}

`stall` in the profile: the time from the last change of the GOOD value
to the last GOOD sample of the window. A long stall on a measurement suggests a
frozen value.

## Stepped {#stepped}

The fourth field of the sampling contract. `stepped yes` reads the
value as holding until the next sample, the way a historian stores a
value it records only on change. It lets a compression gap be told from
a data-loss gap.

## Switchback {#switchback}

A randomized trial of two settings on one unit: the unit alternates
between A and B in blocks, in an order drawn at random before the
trial, and the difference B - A is tested against every order the draw
could have given.

## T2 {#t2}

Hotelling's T-squared: how far a row sits from the baseline's center
inside the plane the PCA model keeps, in units of the baseline's own
variation. A move the correlations allow, only bigger, shows in T2.

## Tag {#tag}

One measured or computed signal in a historian, such as a flow or a
temperature, with its own name, unit and history. tsdive keeps one tag
per archive.

## Time constant {#time-constant}

The time a first-order process takes to cover 63% of a step. After 3
time constants it has covered 95%, which is the washout a switchback
block needs.

## Valid {#valid}

A sample is valid when its severity is GOOD and its value is a finite
number. `valid 0.998` in the profile is the share of rows that are
valid.

## Washout {#washout}

The start of every switchback block that the analysis leaves out, while
the process settles after the switch. 15 min in the demo trial, three
time constants of its 5-minute lag.

## Window {#window}

The span of time a command reads, `START/END` in ISO 8601, such as
`2024-03-30T20:00:00Z/2024-03-31T06:00:00Z`. Both ends are included. A
window is a time interval, not a rolling-window size.
