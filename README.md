# Cross-Document Event Co-referencing

## Task

The task at hand is to fine-tune a small language model (RoBERTa) to improve performance on a cross-document event coreference task. The goal is to enable the model to correctly predict whether two event mentions in a sentence pair refer to the same real-world event.

## Installation and Setup

All dependencies can be found in `requirements.txt`.

## Usage

Code can be run by simply clicking the run button on each file, or running `python roberta_ft.py` in the terminal. Training will run first, followed by evaluation, and the metrics will be printed afterwards.

## Data

Some sample data has been provided in `sample_data`, and each line follows this format and is tab-separated:

- Sentence 1
- Start token index of event1 trigger word
- End token index of event1 trigger word(inclusive)
- Start token index of event1 participant phrase 1
- End token index of event1 participant phrase 1(inclusive)
- Start token index of event1 participant phrase 2
- End token index of event1 participant phrase 2(inclusive)
- Start token index of event1 time
- End token index of event1 time (inclusive)
- Start token index of event1 location
- End token index of event1 location (inclusive)
- Sentence 2
- Start token index of event2 trigger word
- End token index of event2 trigger word (inclusive)
- Start token index of event2 participant phrase 1
- End token index of event2 participant phrase 1 (inclusive)
- Start token index of event2 participant phrase 2
- End token index of event2 participant phrase 2 (inclusive)
- Start token index of event2 time
- End token index of event2 time (inclusive)
- Start token index of event2 location
- End token index of event2 location (inclusive)
- label: A binary label indicating whether the events are coreferent (1) or not (0).

*Note: index -1 means this information is not provided in data.
