import xml.etree.ElementTree as ET
import math
from typing import List, Dict, Optional, Any

# Define the possible labels as provided
POSSIBLE_LABELS = [
    "lack of specific details",
    "user intent unclear",
    "normal query",
    "out of scope requests",
    "multiple issue in single message"
]

def calculate_multitoken_label_confidence(
    response: Any, # Expects an OpenAI ChatCompletion response object
    label_tag_name: str = "current_turn_label",
    possible_labels: List[str] = POSSIBLE_LABELS
) -> Optional[Dict[str, Any]]:
    """
    Calculates the confidence of a multi-token label within an XML structure
    using the Average Log Probability method.

    Args:
        response: The response object from the OpenAI Chat Completions API call.
                  Must have been called with logprobs=True.
        label_tag_name: The name of the XML tag containing the label.
        possible_labels: A list of the expected string values for the label.

    Returns:
        A dictionary containing:
        - 'label': The extracted label string.
        - 'confidence_avg_logprob': The average log probability of the tokens
                                     constituting the label. Higher (less negative)
                                     values indicate higher confidence.
        - 'num_tokens': The number of tokens the label consists of.
        Returns None if the label cannot be found, parsed, or aligned with logprobs.
    """
    try:
        # 1. Extract XML content and logprobs list from response
        if not response.choices or not response.choices[0].message or not response.choices[0].logprobs:
            print("Error: Response object structure is invalid or missing logprobs.")
            return None

        xml_content_string = response.choices[0].message.content
        logprob_content_list = response.choices[0].logprobs.content # List of token logprob info

        if not xml_content_string or not logprob_content_list:
             print("Error: Missing XML content or logprob list in the response.")
             return None

        # 2. Parse XML to find the label value
        root = ET.fromstring(xml_content_string)
        label_element = root.find(label_tag_name)

        if label_element is None or label_element.text is None:
            print(f"Error: XML tag '<{label_tag_name}>' not found or is empty.")
            return None

        extracted_label = label_element.text.strip()

        # Optional: Validate if the extracted label is one of the expected ones
        if extracted_label not in possible_labels:
            print(f"Warning: Extracted label '{extracted_label}' not in predefined possible labels.")
            # Decide if you want to proceed or return None here based on strictness
            # return None

        # 3. Align the extracted label string with the token sequence
        # This is the most complex part due to potential tokenization differences.
        # We'll iterate through the token list and try to find a contiguous
        # sequence whose concatenated text matches the extracted label.

        label_tokens_info = []
        found_match = False
        current_index = 0
        
        # Clean the target label for comparison
        target_label_clean = extracted_label.replace(" ", "").lower() 

        while current_index < len(logprob_content_list):
            potential_match_tokens = []
            concatenated_token_text = ""
            
            # Try to build a match starting from current_index
            for j in range(current_index, len(logprob_content_list)):
                token_info = logprob_content_list[j]
                potential_match_tokens.append(token_info)
                # Concatenate token text, stripping individual token whitespace 
                # and making lowercase for robust matching
                concatenated_token_text += token_info.token.strip() 

                # Check if the cleaned, concatenated string matches the cleaned label
                if concatenated_token_text.lower().replace(" ", "") == target_label_clean:
                    label_tokens_info = potential_match_tokens
                    found_match = True
                    break # Found the sequence
            
            if found_match:
                break # Exit outer loop
            
            # If no match started at current_index, advance index
            current_index += 1


        if not found_match:
            print(f"Error: Could not align extracted label '{extracted_label}' with tokens in logprobs.")
            # This can happen if tokenization is complex (e.g., includes unexpected whitespace tokens)
            # or if the label text differs slightly from the token concatenation.
            return None

        # 4. Calculate Average Log Probability
        if not label_tokens_info: # Should not happen if found_match is True, but safety check
             print("Error: Token alignment succeeded but token list is empty.")
             return None
             
        sum_logprobs = sum(token.logprob for token in label_tokens_info)
        num_tokens = len(label_tokens_info)
        avg_logprob = sum_logprobs / num_tokens

        return {
            "label": extracted_label,
            "confidence_avg_logprob": avg_logprob,
            "num_tokens": num_tokens
        }

    except ET.ParseError:
        print(f"Error: Failed to parse XML content.")
        return None
    except AttributeError as ae:
         print(f"Error: Issue accessing attributes in the response object. Is logprobs=True set? Error: {ae}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Example Usage Placeholder ---
# Assuming 'client' is an initialized OpenAI client and 'my_prompt' is defined

# try:
#     # Ensure logprobs=True is set in the API call
#     response = client.chat.completions.create(
#         model="gpt-4o-mini", # Or your chosen model supporting logprobs
#         messages=my_prompt,
#         temperature=0, # Lower temperature for more deterministic output/labels
#         logprobs=True
#     )

#     confidence_data = calculate_multitoken_label_confidence(response)

#     if confidence_data:
#         print(f"Extracted Label: {confidence_data['label']}")
#         print(f"Number of Tokens: {confidence_data['num_tokens']}")
#         print(f"Average Log Probability: {confidence_data['confidence_avg_logprob']:.4f}")

#         # Example of applying a threshold (needs empirical tuning!)
#         # Note: Average Logprob is negative. Less negative means more confident.
#         # A threshold might be, e.g., -0.5. Adjust based on validation.
#         avg_logprob_threshold = -0.5

#         is_ambiguous = confidence_data['label'] != "normal query"
#         confidence_score = confidence_data['confidence_avg_logprob']

#         final_decision = confidence_data['label']

#         if is_ambiguous and confidence_score < avg_logprob_threshold:
#             print(f"Low confidence ({confidence_score:.4f} < {avg_logprob_threshold}) for ambiguous label '{confidence_data['label']}'.")
#             # Option 1: Override label (as discussed in context)
#             # final_decision = "normal query"
#             # print("Overriding to 'normal query'.")
#             # Option 2: Flag for clarification (recommended in context)
#             print("Recommend triggering clarification dialogue.")
#         else:
#             print(f"Sufficient confidence ({confidence_score:.4f}) or label is 'normal query'.")

#         print(f"Final effective label/action based on confidence: {final_decision}")


#     else:
#         print("Could not calculate confidence.")

# except Exception as e:
#     print(f"API call or processing failed: {e}")