import streamlit as st
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from decimal import Decimal, getcontext  # Import Decimal for precise floating-point arithmetic

# Set page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Chatbot Test Results", page_icon="🤖", layout="wide")

# Import chatbot function AFTER setting page config
from chatbot import get_bot_response  # Import your chatbot's response function

# Load the test set
def load_test_set():
    test_set_path = "testset.csv.txt"
    test_set = pd.read_csv(test_set_path, header=None, names=["question", "expected_answer"])
    return test_set

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]  # Return the first (and only) score dictionary

# Function to format scores for better readability
def format_score(score):
    # Set the precision for Decimal to handle extremely small values
    getcontext().prec = 10  # Adjust precision as needed
    decimal_score = Decimal(str(score))  # Convert score to Decimal for precise comparison

    if decimal_score < Decimal('1e-10'):  # If the score is extremely small, display it in exponential format
        return f"{decimal_score:.2e}"  # Exponential format (e.g., 1.41e-50)
    else:
        return f"{decimal_score:.4f}"  # Otherwise, format to 4 decimal places

# Initialize session state for tracking the current question
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "results" not in st.session_state:
    st.session_state.results = []

if "average_scores" not in st.session_state:
    st.session_state.average_scores = {"bleu": 0, "rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

if "stop_testing" not in st.session_state:
    st.session_state.stop_testing = False  # Flag to stop testing

# Load the test set
test_set = load_test_set()

# Streamlit UI
def main():
    st.title("🤖 Chatbot Automation Testing")

    # Add a Stop Button
    if st.button("Stop Testing"):
        st.session_state.stop_testing = True  # Set the flag to stop testing

    # Display the current question and results
    if not st.session_state.stop_testing and st.session_state.current_index < len(test_set):
        question = test_set.iloc[st.session_state.current_index]["question"]
        expected_answer = test_set.iloc[st.session_state.current_index]["expected_answer"]

        st.subheader(f"Test Case {st.session_state.current_index + 1}")
        st.write(f"**Question:** {question}")
        st.write(f"**Expected Answer:** {expected_answer}")

        # Get the chatbot's response
        bot_response = get_bot_response(question)
        st.write(f"**Bot Response:** {bot_response}")

        # Calculate BLEU score
        bleu_score = calculate_bleu(expected_answer, bot_response)
        st.write(f"**BLEU Score:** {format_score(bleu_score)}")  # Use formatted score

        # Calculate ROUGE scores
        rouge_score = calculate_rouge(expected_answer, bot_response)
        st.write(f"**ROUGE-1 F1 Score:** {format_score(rouge_score['rouge-1']['f'])}")  # Use formatted score
        st.write(f"**ROUGE-2 F1 Score:** {format_score(rouge_score['rouge-2']['f'])}")  # Use formatted score
        st.write(f"**ROUGE-L F1 Score:** {format_score(rouge_score['rouge-l']['f'])}")  # Use formatted score

        # Store results with test case number
        st.session_state.results.append({
            "Test Case": st.session_state.current_index + 1,  # Add test case number
            "Question": question,
            "Expected Answer": expected_answer,
            "Bot Response": bot_response,
            "BLEU Score": bleu_score,
            "ROUGE-1 F1": rouge_score["rouge-1"]["f"],
            "ROUGE-2 F1": rouge_score["rouge-2"]["f"],
            "ROUGE-L F1": rouge_score["rouge-l"]["f"],
        })

        # Update average scores
        st.session_state.average_scores["bleu"] += bleu_score
        st.session_state.average_scores["rouge-1"] += rouge_score["rouge-1"]["f"]
        st.session_state.average_scores["rouge-2"] += rouge_score["rouge-2"]["f"]
        st.session_state.average_scores["rouge-l"] += rouge_score["rouge-l"]["f"]

        # Move to the next question
        st.session_state.current_index += 1

        # Automatically rerun to process the next test case
        st.rerun()

    # Display final results and average scores if testing is stopped or completed
    if st.session_state.stop_testing or st.session_state.current_index >= len(test_set):
        st.subheader("Final Results")

        # Convert results to a DataFrame
        results_df = pd.DataFrame(st.session_state.results)

        # Reorder columns to place "Test Case" first
        results_df = results_df[["Test Case", "Question", "Expected Answer", "Bot Response", "BLEU Score", "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"]]

        # Remove the default index column (left-side indexing row)
        results_df.set_index("Test Case", inplace=True)

        # Format BLEU scores in exponential format for extremely small values
        results_df["BLEU Score"] = results_df["BLEU Score"].apply(lambda x: format_score(x))

        # Display the DataFrame without the default index
        st.dataframe(results_df)

        # Calculate and display average scores
        num_cases = len(st.session_state.results)  # Use the number of completed cases
        if num_cases > 0:
            average_bleu = st.session_state.average_scores["bleu"] / num_cases
            average_rouge_1 = st.session_state.average_scores["rouge-1"] / num_cases
            average_rouge_2 = st.session_state.average_scores["rouge-2"] / num_cases
            average_rouge_l = st.session_state.average_scores["rouge-l"] / num_cases

            st.subheader("Average Scores")
            st.write(f"**Average BLEU Score:** {format_score(average_bleu)}")  # Use formatted score
            st.write(f"**Average ROUGE-1 F1 Score:** {format_score(average_rouge_1)}")  # Use formatted score
            st.write(f"**Average ROUGE-2 F1 Score:** {format_score(average_rouge_2)}")  # Use formatted score
            st.write(f"**Average ROUGE-L F1 Score:** {format_score(average_rouge_l)}")  # Use formatted score

        # Add a download button for the results
        csv = results_df.to_csv(index=True).encode('utf-8')  # Include the "Test Case" index in the CSV
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="chatbot_test_results.csv",
            mime="text/csv",
        )

# Run the Streamlit app
if __name__ == "__main__":
    main()
