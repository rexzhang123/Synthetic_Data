import replicate
import random
import logging


# Configuration
# DO NOT CHANGE THESE PARAMETERS
MODEL_NAME = "mistralai/mistral-7b-v0.1:3e8a0fb6d7812ce30701ba597e5080689bef8a013e5c6a724fafb108cc2426a0"
MAX_NEW_TOKENS = 1024

# YOU CAN CHANGE THESE PARAMETERS
NUM_CONVERSATIONS = 50
QUALITY_THRESHOLD = 0

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Cache structure: {"conversation": String, "quality": Integer}
conversation_quality_cache = {}
conversations = []


def run_mistral(prompt, max_new_tokens=MAX_NEW_TOKENS):
    try:
        output = replicate.run(
            MODEL_NAME,
            input={"prompt": prompt, "max_new_tokens": max_new_tokens}
        )
        # HINT: See if you want to change this to keep only the first output. What are the benefits / drawbacks?
        return "".join(list(output))
    except Exception as e:
        logging.error(f"API call failed with error: {e}")
        return ""


def check_conversation_quality(conversation):
    # Get quality from cache or set default to None
    quality = conversation_quality_cache.get(conversation, {}).get("quality")

    if quality is None:
        # TODO: Write a better, more detailed quality prompt.
        # quality_prompt = (f"Your task is to rate te quality of the following conversation on a scale of 1 to 5, with 1 being lowest quality and 5 being  "

        #                   "Rate the quality of the following conversation on a scale of 1 to 5:\n\n{conversation}\n\nRating:")

        quality_prompt = (
            "Your task is to rate the quality of the following conversation on a scale of 1 to 5."
            "Use the following criteria for your rating: \n"
            "1. Truthfulness and Substantiation: The response should be honest and backed by evidence or sound reasoning.\n"
            "2. Relevance: The response should address the user's query directly, omitting irrelevant information.\n"
            "3. Clarity and Avoidance of Ambiguity/Obscurity: The response should be clear and straightforward, avoiding any ambiguity or unnecessarily complex language.\n"
            "4. Information Appropriateness: The response should provide information that is as informative as necessary to address the user's query, but not overly detailed or superfluous.\n"
            "5. Grammar and Completeness: The response should be free of grammatical errors and incomplete sentences.\n\n"
            "A score of '1' indicates a poor response, failing significantly in most or all criteria. A score of '5' indicates an excellent response, meeting many criteria effectively.\n\n"
            "Example of a '1' Score Response:\n"
            "- USER: What causes rainbows?\n"
            "- ASSISTANT: Rainbows are magic, you see them when birds paint the sky.\n"
            "Explanation: This response is factually incorrect (violates criterion 1), irrelevant and whimsical (violates criterion 2), and though clear, it's misleading and lacks necessary "
            "information (violates criteria 3 and 4). It is grammatically correct, but this is insufficient to raise the overall quality (meets criterion 5 minimally).\n\n"
            "Example of a '5' Score Response:\n"
            "- USER: What causes rainbows?\n"
            "- ASSISTANT: Rainbows are caused by the refraction, dispersion, and reflection of light in water droplets, resulting in a spectrum of light appearing in the sky. "
            "They take the form of a multicoloured circular arc. Rainbows caused by sunlight always appear in the section of sky directly opposite the sun.\n"
            "Explanation: This response is accurate and evidence-based (meets criterion 1), directly relevant to the user's question (meets criterion 2), "
            "clear and precise (meets criterion 3), provides comprehensive information without unnecessary details (meets criterion 4), and is grammatically well-structured (meets criterion 5).\n\n"
            "Rate the following conversation: " + conversation + "\n\n"
            "Rating (1-5) where 1 is worst and 5 is best: "
        )

        rating = run_mistral(quality_prompt)

        try:
            score = int(rating.strip().split()[-1])
            quality = min(max(score, 1), 5)

            count_sorry = conversation.lower().count('sorry')
            is_sorry_twice_or_more = count_sorry >= 2

            if is_sorry_twice_or_more:
                quality = 1
        except:
            quality = 3
        conversation_quality_cache[conversation] = {"quality": quality}

    return quality


def generate_conversation():
    '''
    The goal of this function is to generate a different conversation each time.
    TODO: In this function edit the code to make sure that the conversation block as at most 6 assistant messages.
    TODO: If it has more than 6 blocks, only use the first 6. 
    '''
    system_message = ("<SYS> You are an AI assistant that generates conversations. Your task is generate a conversation between a user and an AI assistant."
                      " The conversation must be between 1 to 6 turns, with each turn comprising of 1 user request followed by 1 assistant response. The conversation should span a variety of subjects."
                      " Your response format must be: USER: <user message> ASSISTANT: <assistant message>"
                      " </SYS>")

    user_message = ("Your task is to generate a conversation between a user and AI assistant. "
                    "The conversation must be between 1 to 6 turns, with one turn comprising of 1 user request followed by 1 assistant response.  "
                    "The conversation can span any subject, including technology, lifestyle, education, entertainment, personal advice, science, "
                    "health, culture, coding, creative writing, or any other topic. Aim to vary the topics with each new prompt execution. "
                    "The conversation may involve different types of interactions such as providing information, offering advice, brainstorming, teaching, problem-solving, creative storytelling, or engaging in polite small talk. "
                    "Ensure your response is informative, engaging, clear, and grammatically correct. It should be empathetic, sophisticated, and knowledgeable. "
                    "The conversation must be contextually relevant and personalized based on the user's message. Your response format MUST be: USER: <user message> ASSISTANT: <assistant message>"
                    )

    prompt = f"{system_message}\nUSER: {user_message}\nASSISTANT:"
    conversation = run_mistral(prompt)
    print(conversation)

    assistant_messages = conversation.split('ASSISTANT:')[:7]

    # Join these parts back together with 'ASSISTANT:'
    conversation = 'ASSISTANT:'.join(assistant_messages)

    # Store in cache (with quality uninitialized)
    conversation_quality_cache[conversation] = {}
    return conversation


def main():
    logging.info("Generating conversations...")

    # conversations is a global variable
    conversations = [generate_conversation() for _ in range(NUM_CONVERSATIONS)]

    # HINT: You should save the generated conversations to a file, so you can then inspect them, filter them etc.
    # with open('raw_conversations.txt', 'w') as file:
    #     # Write each list item on a new line
    #     for item in conversations:
    #         file.write(f"{item}\n")


    # logging.info("Filtering by quality...")
    # high_quality_conversations = [conv for conv in conversations if
    #                               int(check_conversation_quality(conv)) >= QUALITY_THRESHOLD]
    
    # with open('my_dict.txt', 'w') as file:
    #     # Write each dictionary item, value first then key
    #     for key, value in conversation_quality_cache.items():
    #         file.write(f"{value}: {key}\n")

    # HINT: You should save the quality dict (`conversation_quality_cache`) and see if the ratings are what you want.
    with open("synthetic_dataset.txt", "a+") as f:
        # for conv in high_quality_conversations:
        for conv in conversations:
            conversation_formatted = conv.replace("\n",
                                                  "\\n")  # escape newlines, so you can keep your result on one line in the output

            # removes everything before the first "USER:"
            userIdx = conversation_formatted.find('USER:')
            if userIdx == -1:
                continue
            else:
                conversation_formatted = conversation_formatted[userIdx:]

            f.write(conversation_formatted + "\n")

    # print(conversation_quality_cache)

    # logging.info(
    #     f"Saved {len(high_quality_conversations)} high-quality conversations.")


if __name__ == "__main__":
    main()
