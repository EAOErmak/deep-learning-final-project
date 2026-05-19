# Final Project: Proposal, Weekly Progress, and Final Submission

**Related to**: Applied Deep Learning Project  
**Estimated time**: 1 month  
**Work mode**: Individual or team project, depending on instructor approval

---

## Goal

In this project, you will choose a real dataset, define a deep learning problem, train and evaluate a model, and document your work in a GitHub repository.

The project should show that you can:

- choose a realistic machine learning task,
- understand and clean a real dataset,
- build a baseline model,
- improve the model using deep learning methods from the course,
- evaluate results with appropriate metrics,
- explain your decisions and limitations,
- use GitHub consistently over several weeks.

---

## Topic and Dataset Requirements

Students must choose a topic based on a real dataset.

Allowed dataset sources include:

- Kaggle datasets,
- Hugging Face Datasets,
- UCI Machine Learning Repository,
- Papers with Code datasets,
- official open datasets from governments, research labs, or organizations,
- a student-collected dataset, only if approved by the instructor.

The dataset must be suitable for one of the following deep learning task types:

- image classification,
- object detection or segmentation,
- text classification,
- named entity recognition,
- sequence classification,
- translation or sequence-to-sequence modeling,
- time series forecasting or classification,
- recommendation or ranking,
- another deep learning task approved by the instructor.

The dataset must be large and meaningful enough for training and evaluation. Very small toy datasets are not allowed unless they are part of a carefully designed comparison or prototype approved by the instructor.

Students must cite the dataset source clearly in the proposal and final report.

---

## Repository Requirement

Each student or team must copy the project repository and work inside their own GitHub repository.

The repository must include:

- source code,
- notebook or scripts for experiments,
- dataset loading instructions,
- model training code,
- evaluation code,
- weekly progress reports,
- final report,
- README with instructions for running the project.

Recommended repository structure:

```text
project-repo/
+-- README.md
+-- data/
|   +-- README.md
+-- notebooks/
+-- src/
+-- reports/
|   +-- week-01.md
|   +-- week-02.md
|   +-- week-03.md
|   +-- week-04.md
+-- results/
+-- requirements.txt
+-- final-report.md
```

Large dataset files should not be committed to GitHub. Instead, include download instructions or scripts.

---

## GitHub Commit Requirement

The project lasts one month. Students must make progress every week.

Minimum GitHub activity:

- at least 3 meaningful commits per week,
- commits spread across the week, not all at the end,
- clear commit messages,
- weekly reports committed to the repository,
- no single final upload containing the whole project.

Examples of meaningful commits:

- add dataset loading code,
- clean and inspect dataset,
- add baseline model,
- add training loop,
- add evaluation metrics,
- add visualizations,
- update weekly report,
- fix a model bug and explain the fix,
- improve README instructions.

Examples of weak commits:

- `update`,
- `final`,
- `changes`,
- uploading all files only once near the deadline,
- committing generated cache files or large raw datasets.

Students who do not show weekly GitHub progress may lose points even if the final code runs.

---

## Proposal Requirements

Submit a project proposal before starting the main implementation.

The proposal should be 1-2 pages and include the following sections.

### 1. Project Title

Give a clear and specific title.

Examples:

- CNN Classification of Plant Leaf Diseases
- Sentiment Classification of Airline Reviews
- Named Entity Recognition for News Articles
- Time Series Forecasting for Electricity Demand

### 2. Problem Statement

Explain the task in plain language.

Answer:

1. What problem are you trying to solve?
2. Why is this problem useful or interesting?
3. What will the model predict or generate?

### 3. Dataset

Describe the dataset.

Include:

- dataset name,
- dataset source link,
- number of examples,
- input features,
- target labels or outputs,
- data format,
- license or usage notes if available.

If the dataset is from Kaggle or Hugging Face, include the exact dataset page link.

### 4. Planned Method

Describe the model or models you plan to use.

Include:

- a simple baseline,
- one deep learning model,
- loss function,
- evaluation metrics,
- train/validation/test split plan.

Examples:

- Baseline: logistic regression or small MLP
- Deep learning model: CNN, LSTM, GRU, Transformer, pretrained model fine-tuning
- Metrics: accuracy, F1-score, precision, recall, RMSE, MAE, BLEU, confusion matrix

### 5. Expected Challenges

Write 3-5 sentences about possible difficulties.

Examples:

- dataset imbalance,
- noisy labels,
- small dataset size,
- overfitting,
- long training time,
- missing values,
- unclear evaluation metric,
- GPU or memory limitations.

### 6. Weekly Plan

Include a one-month plan.

Use this format:

| Week | Planned Work | Expected Output |
| --- | --- | --- |
| Week 1 | Dataset selection, repository setup, exploratory data analysis | Proposal, README, dataset summary |
| Week 2 | Data preprocessing, train/validation/test split, baseline model | Baseline results and Week 2 report |
| Week 3 | Deep learning model training and experiments | Model results, plots, error analysis |
| Week 4 | Improvements, final evaluation, final report and presentation | Final code, final report, slides/demo |

---

## Weekly Report Requirements

Each week, submit a short progress report in the repository under `reports/`.

Each weekly report should include:

- what was completed this week,
- important commits or files changed,
- experiments run,
- results so far,
- problems or blockers,
- plan for next week.

Recommended file names:

```text
reports/week-01.md
reports/week-02.md
reports/week-03.md
reports/week-04.md
```

Each report should be approximately 0.5-1 page.

---

## Final Submission Requirements

At the end of the month, submit the GitHub repository link and final report.

The final repository must include:

- working code or notebook,
- clear README,
- dataset instructions,
- preprocessing code,
- model code,
- training code,
- evaluation code,
- weekly reports,
- final report,
- saved figures or result tables,
- requirements file or environment instructions.

The final report should include:

1. Project title
2. Problem statement
3. Dataset description
4. Data preprocessing
5. Model architecture
6. Training setup
7. Evaluation metrics
8. Results table
9. Error analysis
10. Limitations
11. Conclusion
12. References

---

## Final Presentation or Demo

Students will give a short presentation or demo.

Recommended length: 5-7 minutes.

The presentation should include:

- problem and dataset,
- model approach,
- main results,
- one interesting success case,
- one failure case,
- what you would improve next.

During the project defense, students must also answer questions from the exam papers. The exam/defense question part is worth 20 points.

Quizzes will be based on each student's project topic and the course materials already covered in class.

---

## Proposal Rubric

Total: 4 points

| Category | Points | Criteria |
| --- | ---: | --- |
| Topic clarity | 1 | Problem is specific, understandable, and appropriate for deep learning |
| Dataset choice | 1 | Dataset is real, clearly sourced, usable, and relevant to the task |
| Problem formulation | 1 | Inputs, outputs, and prediction target are clearly defined |
| Planned method and weekly plan | 1 | Baseline, deep learning model, metrics, split plan, and weekly milestones are reasonable |

---

## Weekly Progress Rubric

Total: 4 points

| Category | Points | Criteria |
| --- | ---: | --- |
| GitHub commits | 2 | At least 3 meaningful commits per week, spread across the project period |
| Weekly reports | 1 | Four reports are submitted and describe progress, results, blockers, and next steps |
| Repository organization | 1 | Files are organized clearly with README, reports, code, and results folders |

---

## Final Project Rubric

Total: 12 points

| Category | Points | Criteria |
| --- | ---: | --- |
| Dataset understanding and preprocessing | 2 | Dataset is loaded correctly, inspected, cleaned, split properly, and explained |
| Baseline model | 1 | A simple baseline is implemented and evaluated for comparison |
| Deep learning model | 3 | Model is appropriate, implemented correctly, and connected to course concepts |
| Training and experiments | 2 | Training loop, hyperparameters, loss curves, and experiment choices are documented |
| Evaluation and results | 2 | Metrics are appropriate, results are reported clearly, and comparisons are fair |
| Error analysis and interpretation | 1 | Student analyzes mistakes, limitations, and possible improvements |
| Code quality, final report, and presentation | 1 | Code is readable, reproducible, and final explanation is clear |

---

## Project Work Grade

| Component | Points |
| --- | ---: |
| Project proposal | 4 |
| Weekly GitHub progress and reports | 4 |
| Final project implementation, report, and presentation | 12 |
| **Project work total** | **20** |

---

## Total Grade

| Component | Points |
| --- | ---: |
| Project work | 20 |
| Exam/defense questions from exam papers | 20 |
| **Total** | **40** |

During the project defense, students must be ready to answer questions connected to the exam papers, their own project, and the course materials already covered.

Quizzes will be based on the student's project direction and passed course materials.

---

## Important Rules

- Use a real dataset.
- Cite the dataset source.
- Do not use the test set for model selection.
- Do not commit large raw datasets to GitHub.
- Do not submit a project copied from the internet without understanding it.
- All code must be runnable or clearly explained.
- All team members must contribute commits if the project is done in a team.
- Late, empty, or last-minute repositories may lose points even if the final report looks complete.
