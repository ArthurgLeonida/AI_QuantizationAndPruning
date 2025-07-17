from evaluate import load
import collections
import numpy as np
from tqdm.auto import tqdm # For progress bar during post-processing

squad_metric = load("squad_v2")

def postprocess_qa_predictions(
    examples, features, predictions, tokenizer, 
    n_best_size=20, max_answer_length=30, 
    no_answer_threshold=0.0
):
    all_start_logits, all_end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # Modified: all_predictions will now store dictionaries, not just text
    all_predictions = collections.OrderedDict() 

    print("\n--- Starting SQuAD v2.0 Post-processing ---")

    for example_index, example in enumerate(tqdm(examples, desc="Post-processing predictions")):
        feature_indices = features_per_example[example_index]

        min_null_score = None 
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            input_ids = features[feature_index]["input_ids"]

            # SQuAD v2.0 specific: Score for "no answer" (CLS token at index 0)
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # --- Derive context boundaries from input_ids and tokenizer's SEP token ---
            try:
                first_sep_token_index = input_ids.index(tokenizer.sep_token_id)
            except ValueError: 
                first_sep_token_index = -1 

            try:
                second_sep_token_index = len(input_ids) - 1 - input_ids[::-1].index(tokenizer.sep_token_id)
            except ValueError: 
                second_sep_token_index = first_sep_token_index 

            context_start_token_idx = first_sep_token_index + 1
            
            if second_sep_token_index == first_sep_token_index:
                context_end_token_idx = len(input_ids) - 1
                while context_end_token_idx >= 0 and offset_mapping[context_end_token_idx][0] is None:
                    context_end_token_idx -= 1
            else:
                context_end_token_idx = second_sep_token_index - 1
            
            context_start_token_idx = max(context_start_token_idx, 0)
            context_end_token_idx = min(context_end_token_idx, len(input_ids) - 1)
            # --- End context boundary derivation ---
            
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Filter invalid spans (existing logic)
                    if not (context_start_token_idx <= start_index <= context_end_token_idx and \
                            context_start_token_idx <= end_index <= context_end_token_idx):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    start_char_offset = offset_mapping[start_index][0]
                    end_char_offset = offset_mapping[end_index][1]
                    if start_char_offset is None or end_char_offset is None:
                        continue
                    actual_context_start_char = offset_mapping[context_start_token_idx][0]
                    actual_context_end_char = offset_mapping[context_end_token_idx][1]
                    if not (start_char_offset >= actual_context_start_char and end_char_offset <= actual_context_end_char):
                         continue

                    predicted_answer_text = context[start_char_offset:end_char_offset]
                    
                    valid_answers.append(
                        {
                            "text": predicted_answer_text,
                            "score": start_logits[start_index] + end_logits[end_index],
                        }
                    )
        
        # Select the best answer from the valid spans found
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        # SQuAD v2.0 specific: Compare best span score with NULL score using threshold ---
        final_prediction_text = ""
        no_answer_probability = min_null_score if min_null_score is not None else -float('inf') 

        # The decision now uses the 'no_answer_threshold'
        if no_answer_probability > best_answer["score"] + no_answer_threshold: # <--- CHANGE HERE
            final_prediction_text = "" # Predict empty string for no answer
        else:
            final_prediction_text = best_answer["text"]

        # Store the complete prediction dictionary for this example
        all_predictions[example["id"]] = {
            "prediction_text": final_prediction_text,
            "no_answer_probability": no_answer_probability
        }

    # Format predictions to include 'no_answer_probability' ---
    formatted_predictions = [
        {"id": k, "prediction_text": v["prediction_text"], "no_answer_probability": v["no_answer_probability"]} 
        for k, v in all_predictions.items()
    ]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    
    print("--- SQuAD Post-processing Complete ---")
    return squad_metric.compute(predictions=formatted_predictions, references=references)

# The compute_squad_metrics function remains unchanged from previous version
def compute_squad_metrics(p, *, original_examples, tokenized_features, tokenizer, no_answer_threshold=0.0):
    results = postprocess_qa_predictions(
        examples=original_examples,
        features=tokenized_features,
        predictions=p.predictions,
        tokenizer=tokenizer,
        no_answer_threshold=no_answer_threshold
    )
    return results