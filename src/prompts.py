PROMPTS = {
    "old": '''I am creating a dictionary of terms and ideas that are specific to a company. Use the following documents and context to pull out company-specific information, lingo, and other information that may help in answering a user's question. Make sure you pull out any terms that may be ambiguous or are different from what your training indicates.''',
    "update1": '''I am developing a specialized dictionary focusing on company or peer group-specific terminology and concepts. Utilize the provided documents and context to identify and extract unique or context-specific terms, jargon, and meanings, particularly those that may be ambiguous or differ from standard usage. Ensure these terms are well-defined to assist in addressing user inquiries in future interactions. Refrain from defining terms that you already know and that can be considered common knowledge or can be found in a dictionary.''',
    "update2": '''I am developing a specialized dictionary focusing on company or peer group-specific terminology and concepts. Utilize the provided documents and context to identify and extract unique or context-specific terms, jargon, and meanings, particularly those that may be ambiguous or differ from standard usage. Ensure these terms are well-defined to assist in addressing user inquiries in future interactions. Refrain from defining terms that you already know and that can be considered common knowledge or can be found in a dictionary. If you find terms used in different ways, please use the newer information based on the given documents and context.''',
}

def get_prompt(version: str) -> str:
    return PROMPTS.get(version, PROMPTS["old"])

if __name__ == "__main__":
    for k, v in PROMPTS.items():
        print(f"Prompt '{k}':\n{v}\n---\n")
