from transformers import AutoTokenizer
import os

def get_tokenizer(model_name="distilbert-base-uncased", save_path=None):
    """
    Loads and returns a pre-trained tokenizer.
    If save_path is provided and the tokenizer files exist, it loads from there.
    Otherwise, it loads from the Hugging Face Hub.
    """
    if save_path and os.path.isdir(save_path):
        print(f"Loading tokenizer from local path: {save_path}")
        return AutoTokenizer.from_pretrained(save_path)
    else:
        print(f"Loading tokenizer from Hugging Face Hub: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if save_path:
            print(f"Saving tokenizer to local path: {save_path}")
            tokenizer.save_pretrained(save_path)
        return tokenizer

def prepare_squad_features(examples, tokenizer_path, max_length=512, stride=128):
    """
    Tokenizes SQuAD question and context, and aligns labels (start_positions, end_positions).
    This function is designed to be used with datasets.map() and multiprocessing.
    """

    local_tokenizer = get_tokenizer(model_name="distilbert-base-uncased", save_path=tokenizer_path)

    tokenized_examples = local_tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second", # Truncate only the context (second sequence)
        max_length=max_length,
        stride=stride, # Overlapping context chunks to ensure answers aren't cut
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["example_id"] = []
    
    
    for i in range(len(tokenized_examples["input_ids"])):
        offsets = tokenized_examples["offset_mapping"][i] 
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(local_tokenizer.cls_token_id) # Theoretically will always be the first token
        sequence_ids = tokenized_examples.sequence_ids(i) # 0 for question, 1 for context, -1 for special tokens

        sample_index = tokenized_examples["overflow_to_sample_mapping"][i] # Returns the original example index
        answers = examples["answers"][sample_index]
        question_id = examples["id"][sample_index]

        tokenized_examples["example_id"].append(question_id)

        # If there are no answers, we label it as impossible
        if not answers["text"]:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find the start of the context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1: 
            token_start_index += 1

        # Find the end of the context
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Check if the answer is out of the span
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char): 
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_token_index = token_start_index
            while start_token_index <= token_end_index and offsets[start_token_index][0] <= start_char:
                start_token_index += 1
            tokenized_examples["start_positions"].append(start_token_index - 1)

            end_token_index = token_end_index
            while end_token_index >= token_start_index and offsets[end_token_index][1] >= end_char:
                end_token_index -= 1
            tokenized_examples["end_positions"].append(end_token_index + 1)

    return tokenized_examples

def check_tokenization():
    '''
    This function is used to check the tokenization process for SQuAD examples.
    It will print the original question, context, and answer, along with the tokenized features
    '''
    from datasets import load_dataset

    # Load SQuAD dataset
    squad_dataset_dict = load_dataset("squad")

    # Initialize tokenizer for the main process
    tokenizer_path = "./distilbert_tokenizer_local"
    local_tokenizer_check = get_tokenizer(model_name="distilbert-base-uncased", save_path=tokenizer_path)

    examples_to_check_indices = [0, 10]

    for original_idx in examples_to_check_indices:
        print(f"\n--- Checking Original Example Index: {original_idx} ---")
        original_example = squad_dataset_dict["validation"][original_idx]

        simulated_batch = {k: [v] for k, v in original_example.items()}
        
        processed_features = prepare_squad_features(simulated_batch, tokenizer_path)
        
        # Print original info
        print(f"Original Question: {original_example['question']}")
        print(f"Original Context: {original_example['context']}")
        if original_example['answers'] and original_example['answers']['text']:
            print(f"Original Answer: {original_example['answers']['text'][0]} (Char Start: {original_example['answers']['answer_start'][0]})")
        else:
            print("Original Answer: None (Unanswerable)")

        # Iterate through all features generated from this original example
        for i in range(len(processed_features["input_ids"])):
            print(f"\n  --- Feature {i+1} / {len(processed_features['input_ids'])} (from original example {original_idx}) ---")
            input_ids = processed_features["input_ids"][i]
            start_pos = processed_features["start_positions"][i]
            end_pos = processed_features["end_positions"][i]
            offsets = processed_features["offset_mapping"][i]

            # Convert input_ids back to tokens for readability
            tokens = local_tokenizer_check.convert_ids_to_tokens(input_ids)
            print(f"  Tokens: {tokens}")
            print(f"  Input IDs Length: {len(input_ids)}")
            print(f"  Start Position (token index): {start_pos}")
            print(f"  End Position (token index): {end_pos}")

            # Check if it's an impossible answer
            if start_pos == local_tokenizer_check.cls_token_id and end_pos == local_tokenizer_check.cls_token_id:
                print("  -> This feature is labeled as IMPOSSIBLE (CLS token as answer span).")
                # Check if this matches the original example's answer status
                if original_example['answers'] and original_example['answers']['text']:
                    print("  WARNING: Original example had an answer, but feature is marked impossible!")
            else:
                # Reconstruct the predicted answer string from token positions
                predicted_answer_tokens = tokens[start_pos : end_pos + 1]
                predicted_answer_text = local_tokenizer_check.decode(
                    input_ids[start_pos : end_pos + 1], skip_special_tokens=True
                ).lower()

                print(f"  Predicted Answer Span (tokens): {predicted_answer_tokens}")
                print(f"  Predicted Answer Text: '{predicted_answer_text}'")

                # Get the character span from offsets
                # Ensure start_pos and end_pos are within bounds
                if 0 <= start_pos < len(offsets) and 0 <= end_pos < len(offsets):
                    char_start = offsets[start_pos][0]
                    char_end = offsets[end_pos][1]
                    
                    print(f"  Character Span in original context: ({char_start}, {char_end})")
                    print(f"  Actual text from original context at this span: '{original_example['context'][char_start:char_end]}'")

                    # Compare with original answer
                    if original_example['answers'] and original_example['answers']['text']:
                        original_answer_text = original_example['answers']['text'][0].lower()
                        if predicted_answer_text == original_answer_text:
                            print("  MATCH: Predicted answer text matches original answer text!")
                        else:
                            print(f"  MISMATCH: Predicted '{predicted_answer_text}' vs Original '{original_answer_text}'")
                    else:
                        print("  WARNING: Original example was unanswerable, but feature found an answer!")
                else:
                    print("  ERROR: start_pos or end_pos out of bounds for offsets!")

        print("\n" + "="*80)

#check_tokenization()