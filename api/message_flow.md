Current TA2/TA3 Message Flow
===================

The system is currently compliant with the API, but doesn't really do its internal processing in a manner that is compliant with the API.  Our current behviour is as follows:

```mermaid
sequenceDiagram
participant ta3 as ta3
participant ta2 as ta2
ta3 ->> ta2: SearchSolutionRequest
note right of ta2: See Note 1.
ta2 -->> ta3: SearchSolutionResponse
ta3 ->> ta2: GetSearchSolutionResults
note right of ta2: See Note 2.
ta2 --x ta3: SearchSolutionResult [pipeline_1, RUNNING]
ta2 --x ta3: SearchSolutionResult [pipeline_1, RUNNING]
ta2 --x ta3: SearchSolutionResult [pipeline_1, RUNNING]
ta2 --x ta3: SearchSolutionResult [pipeline_1, COMPLETED]
ta3 ->> ta2: FitSolutionRequest [pipeline_1]
ta2 -->> ta3: FitSolutionResponse [fitted_pipeline_1]
ta3 ->> ta2: GetFitSolutionResults [fitted_pipeline_1]
note right of ta2: See Note 3.
ta2 --x ta3: FitSolutionResult[fitted_pipeline_1, COMPLETED]
ta3 ->> ta2: ScoreSolutionRequest [pipeline_1]
note right of ta2: See Note 4.
ta2 -->> ta3: ScoreSolutionResponse [pipeline_1]
ta3 ->> ta2: GetScoreSolutionResults [pipeline_1]
ta2 --x ta3: ScoreSolutionResult[pipeline_1, RUNNING]
ta2 --x ta3: ScoreSolutionResult[pipeline_1, RUNNING]
ta2 --x ta3: ScoreSolutionResult[pipeline_1, COMLETED]
ta3 ->> ta2: ProduceSolutionRequest [fitted_pipeline_1]
ta2 -->> ta3: ProduceSolutionResponse [fitted_pipeline_1]
ta3 ->> ta2: GetProduceSolutionResults [fitted_pipeline_1]
ta2 --x ta3: ProduceSolutionResult[fitted_pipeline_1, RUNNING]
ta2 --x ta3: ProduceSolutionResult[fitted_pipeline_1, RUNNING]
ta2 --x ta3: ProduceSolutionResult[fitted_pipeline_1, RUNNING]
ta2 --x ta3: ProduceSolutionResult[fitted_pipeline_1, COMPLETED]
```

Note 1:
At this point we kick off the `exline_task` which creates the pipeline and immediately starts to fit it.

Note 2:
We send back `RUNNING` messages until the fit in `exline_task` completes.

Note 3:
Given that the fit was already completed when `exline_task` ran, we immediately return `COMPLETED` on the fit related calls.

Note 4:
The score request takes solution ID as an argument, not a fitted solution ID, which means that it can technically be invoked without the pipeline being fitted.  This is not possible in our current arrangement because we fit as part of search, but it would need to be addressed.

Expected TA2/TA3 Message Flow
=============================

```mermaid
sequenceDiagram
participant ta3 as ta3
participant ta2 as ta2
ta3 ->> ta2: SearchSolutionRequest
ta2 -->> ta3: SearchSolutionResponse
ta3 ->> ta2: GetSearchSolutionResults
ta2 --x ta3: SearchSolutionResult [pipeline_1, RUNNING]
ta2 --x ta3: SearchSolutionResult [pipeline_1, RUNNING]
ta2 --x ta3: SearchSolutionResult [pipeline_2, RUNNING]
ta2 --x ta3: SearchSolutionResult [pipeline_1, COMPLETED]
ta3 ->> ta2: FitSolutionRequest [pipeline_1]
ta2 -->> ta3: FitSolutionResponse [fitted_pipeline_1]
ta2 --x ta3: SearchSolutionResult [pipeline_2, RUNNING]
ta2 --x ta3: SearchSolutionResult [pipeline_2, COMPLETED]
ta3 ->> ta2: GetFitSolutionResults [fitted_pipeline_1]
ta2 --x ta3: FitSolutionResult[fitted_pipeline_1, RUNNING]
ta2 --x ta3: FitSolutionResult[fitted_pipeline_1, RUNNING]
ta2 --x ta3: FitSolutionResult[fitted_pipeline_1, COMPLETED]
ta2 -->> ta3: ScoreSolutionResponse [pipeline_1]
ta3 ->> ta2: GetScoreSolutionResults [pipeline_1]
ta2 --x ta3: ScoreSolutionResult[pipeline_1, RUNNING]
ta2 --x ta3: ScoreSolutionResult[pipeline_1, RUNNING]
ta2 --x ta3: ScoreSolutionResult[pipeline_1, COMLETED]
ta3 ->> ta2: ProduceSolutionRequest [fitted_pipeline_1]
ta2 -->> ta3: ProduceSolutionResponse [fitted_pipeline_1]
ta3 ->> ta2: FitSolutionRequest [pipeline_2]
ta2 -->> ta3: FitSolutionResponse [fitted_pipeline_2]
ta3 ->> ta2: GetProduceSolutionResults [fitted_pipeline_1]
ta2 --x ta3: ProduceSolutionResult[fitted_pipeline_1, RUNNING]
ta2 --x ta3: ProduceSolutionResult[fitted_pipeline_1, RUNNING]
ta2 --x ta3: ProduceSolutionResult[pipeline_1, COMPLETED]
ta3 ->> ta2: GetFitSolutionResults [fitted_pipeline_1]
ta2 --x ta3: FitSolutionResult[fitted_pipeline_2, RUNNING]
ta2 --x ta3: FitSolutionResult[fitted_pipeline_2, RUNNING]
ta2 --x ta3: FitSolutionResult[fitted_pipeline_2, COMPLETED]
ta2 -->> ta3: ScoreSolutionResponse [pipeline_1]
ta3 ->> ta2: GetScoreSolutionResults [pipeline_2]
ta2 --x ta3: ScoreSolutionResult[pipeline_2, RUNNING]
ta2 --x ta3: ScoreSolutionResult[pipeline_2, RUNNING]
ta2 --x ta3: ScoreSolutionResult[pipeline_2, COMLETED]
ta3 ->> ta2: ProduceSolutionRequest [fitted_pipeline_2]
ta2 -->> ta3: ProduceSolutionResponse [fitted_pipeline_2]
ta3 ->> ta2: GetProduceSolutionResults [fitted_pipeline_2]
ta2 --x ta3: ProduceSolutionResult[pipeline_2, RUNNING]
ta2 --x ta3: ProduceSolutionResult[pipeline_2, RUNNING]
ta2 --x ta3: ProduceSolutionResult[pipeline_2, RUNNING]
```