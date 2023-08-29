import re
import string
import sys

import openai

class GPTCommons:
    def __init__(self):
        self.SPECIAL_CHARACTERS = string.punctuation + "“”‘’"
        self.PATTERN = re.compile(r'[\n\s]+')    

    def clean_text(self, text):
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

    def clean_text(self, text):
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

    def get_completion(self, prompt, model, temperature=0):
        """
        Retrieves a completion using the OpenAI ChatCompletion API with the specified model and parameters.

        Args:
        prompt (str): The user's input or prompt for generating the completion.
        model (str): The OpenAI model identifier (e.g., "gpt-3.5-turbo").
        temperature (float, optional): The degree of randomness in the model's output (default is 0).
                                    A higher value makes the output more random, while a lower value makes it more deterministic.

        Returns:
        str: The generated completion text.

        Example:
        >>> user_prompt = "Translate the following English text to French: 'Hello, how are you?'"
        >>> model_id = "gpt-3.5-turbo"
        >>> get_completion(user_prompt, model_id, temperature=0.7)
        'Bonjour, comment ça va ?'
        """
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    def get_arg(self, arg_name, default=None):
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

    def split_into_chunks(self, text, chunk_size=1000, overlap_percentage=1):
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