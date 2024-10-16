import re
import string
import sys
import tiktoken

from openai import OpenAI

class GPTCommons:
    """
    A utility class for common operations with the GPT model.
    """
    def __init__(self, api_key):
        """
        Initializes GPTCommons with predefined constants and sets the API key.

        Args:
        api_key (str): The API key for authenticating with the OpenAI API.

        Constants:
        - SPECIAL_CHARACTERS (str): Punctuation and special characters used for text cleaning.
        - PATTERN (re.Pattern): Compiled regular expression pattern for cleaning text.
        """
        self.SPECIAL_CHARACTERS = string.punctuation + "“”‘’"
        self.PATTERN = re.compile(r'[\n\s]+')
        self.client = OpenAI(api_key=api_key)

    def reduce_to_max_tokens(self, text, max_tokens, gpt_model) -> str:
        """
        Reduces the input text to a maximum number of tokens for the specified OpenAI model.

        Args:
        text (str): The input text to be reduced.
        max_tokens (int): The maximum number of tokens allowed.
        gptmodel (str): The OpenAI model to use for tokenization (default is "gpt-3.5-turbo").

        Returns:
        str: The reduced text.
        """
        if not isinstance(max_tokens, int):
            raise ValueError("max_tokens must be an integer.")

        # Initialize the tokenizer for the specified model
        tokenizer = tiktoken.encoding_for_model(gpt_model)

        # Tokenize the input text
        tokens = tokenizer.encode(text)

        # Truncate the tokens to the maximum allowed number
        truncated_tokens = tokens[:max_tokens]

        # Convert the tokens back to text
        reduced_text = tokenizer.decode(truncated_tokens)

        return reduced_text

    def clean_text(self, text) -> str:
        """
        Cleans a given text by replacing line breaks, consecutive whitespace, and handling special characters.

        Args:
        text (str): The input text to be cleaned.

        Returns:
        str: The cleaned text.

        Example:
        >>> dirty_text = "This is a\ndirty    text!!"
        >>> clean_text(dirty_text)
        'This is a dirty text  '
        """
        # Replace line breaks and consecutive whitespace with a single space
        text = re.sub(self.PATTERN, ' ', text).strip()

        # Handle special characters (replace with spaces or remove them)
        text = ''.join(char if char not in self.SPECIAL_CHARACTERS else ' ' for char in text)

        return text

    def get_chat_completion(self, prompt, model, temperature=0) -> str:
        """
        Retrieves a completion using the OpenAI ChatCompletion API with the specified model and parameters.

        Args:
        prompt (str): The user's input or prompt for generating the completion.
        model (str): The model to use for generating the completion.
        temperature (float): The temperature to use for generating the completion.

        Returns:
        str: The generated completion.
        """
        """
        response = self.client.chat.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature)
        """
        response = self.client.chat.completions.create(
                messages=[
                        {
                            "role": "system",
                            "content": prompt,
                         }
                        ],
                model=model,
                temperature=temperature
                )
        return response.choices[0].message.content.strip()
    
    def get_arg(self, arg_name, default=None) -> str:
        """
        Retrieves the value of a command-line argument by its name from the sys.argv list.

        Args:
        arg_name (str): The name of the command-line argument to retrieve.
        default: The default value to return if the argument is not found (default is None).

        Returns:
        str or default: The value of the specified command-line argument or the default value if not found.

        Example:
        >>> # Assuming the command-line arguments are ['--lang', 'English', '--url', 'example.com']
        >>> get_arg('--lang', 'Spanish')
        'English'
        >>> get_arg('--url', 'localhost')
        'example.com'
        >>> get_arg('--port', 8080)
        8080
        """
        if "--help" in sys.argv:
            print("Usage: python PD_AI_Sum.py [--help] [--lang] [--url] [--ofile]")
            print("Arguments:")
            print("\t--help\t\tHelp\t\tNone")
            print("\t--lang\t\tLanguage\tEnglish")
            print("\t--url\t\tPDF URL\t\tNone")
            print("\t--ofile\t\tOutpout file\trandom_paper.pdf")
            # Add more argument descriptions here as needed
            sys.exit(0)
        try:
            arg_index = sys.argv.index(arg_name)
            arg_value = sys.argv[arg_index + 1]
            return arg_value
        except (IndexError, ValueError):
            return default

    def split_into_chunks(self, text, chunk_size=1000, overlap_percentage=1) -> list[str]:
        """
        Splits a given text into smaller chunks with a specified chunk size and overlap percentage.

        Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The desired size of each chunk (default is 1000).
        overlap_percentage (float, optional): The percentage of overlap between consecutive chunks (default is 1).
                                            A value of 0 means no overlap, and 1 means 100% overlap.

        Returns:
        list of str: A list containing the text chunks.

        Example:
        >>> text = "This is an example text that needs to be split into smaller chunks."
        >>> split_into_chunks(text, chunk_size=20, overlap_percentage=0.5)
        ['This is an example ', ' example text that ne', 'text that needs to be', ' to be split into smal', ' smaller chunks.']
        """
        #split the web content into chunks of 1000 characters
        text = self.clean_text( text)

        # Calculate the number of overlapping characters
        overlap_chars = int(chunk_size * overlap_percentage)

        # Initialize a list to store the chunks
        chunks = []

        # Loop through the text with the overlap
        for i in range(0, len(text), chunk_size - overlap_chars):
            # Determine the end index of the current chunk
            end_idx = i + chunk_size

            # Slice the text to form a chunk
            chunk = text[i:end_idx]

            # Append the chunk to the list
            chunks.append(chunk)

        return chunks