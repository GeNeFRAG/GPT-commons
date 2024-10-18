import re
import string
import sys
import tiktoken
import tomli

from openai import OpenAI

class GPTCommons:
    """
    A utility class for common operations with the GPT model.

    This class provides various methods to interact with the GPT model, including:
    - Generating chat completions.
    - Handling command-line arguments dynamically.
    - Managing model configurations such as temperature and model type.

    Methods:
    - get_chat_completion(prompt): Generates a chat completion based on the provided prompt.
    - get_arg(arg_name, arg_descriptions, default=None): Retrieves the value of a command-line argument by its name.
    - get_gptmodel(): Returns the current GPT model configuration.
    - get_temperature(): Returns the current temperature setting for the model.
    - reduce_to_max_tokens(text): Reduces the input text to a maximum number of tokens.
    - clean_text(text): Cleans the input text by removing special characters and extra whitespace.
    - split_into_chunks(text, chunk_size, overlap_percentage): Splits the input text into smaller chunks.

    Usage:
    This class can be used in different scripts to standardize interactions with the GPT model and handle common tasks efficiently.
    """
    def __init__(self, api_key, gptmodel, maxtokens, temperature, organization):
        """
        Initializes GPTCommons with predefined constants and sets the API key.

        Args:
        api_key (str): The API key for authenticating with the OpenAI API.
        gptmodel (str): The OpenAI model to use for tokenization.
        maxtokens (int): The maximum number of tokens allowed.
        temperature (float): The temperature to use for generating the completion.
        organization (str): The organization ID for OpenAI.

        Attributes:
        SPECIAL_CHARACTERS (str): Punctuation and special characters used for text cleaning.
        PATTERN (re.Pattern): Compiled regular expression pattern for cleaning text.
        client (OpenAI): The OpenAI client initialized with the provided API key.
        """
        self.SPECIAL_CHARACTERS = string.punctuation + "“”‘’"
        self.PATTERN = re.compile(r'[\n\s]+')
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.gptmodel = gptmodel
        self.maxtokens = maxtokens
        self.temperature = temperature
        self.organization = organization

    @staticmethod
    def initialize_gpt_commons(configfile) -> 'GPTCommons':
        """
        Initializes the GPTCommons class using configuration settings from a TOML file.

        This method reads the OpenAI API key, model, organization, max tokens, and temperature from a specified TOML configuration file and initializes an instance of the GPTCommons class with these settings.

        Args:
        configfile (str): The path to the TOML configuration file containing the OpenAI settings.

        Returns:
        GPTCommons: An instance of the GPTCommons class initialized with the settings from the configuration file.

        Raises:
        KeyError: If any mandatory configuration setting is missing in the TOML file.
        ValueError: If any configuration setting has an invalid value.
        Exception: If there is an error reading the TOML file.
        """
        # Attempt to read the OpenAI API keys and organization from the configuration file
        try:
            with open(configfile, "rb") as f:
                data = tomli.load(f)
        except Exception as e:
            # Print error message and exit if the configuration file cannot be read
            print(f"Error: Unable to read openai.toml file.")
            print(e)
            sys.exit(1)

        try:
            # Extract the API key from the configuration data
            api_key = data["openai"]["apikey"]
            if not api_key:
                raise ValueError("API key is missing or empty in the configuration.")
        except KeyError:
            # Raise an error if the API key is missing in the configuration
            raise KeyError("API key is mandatory and missing in the configuration.")

        try:
            # Extract the GPT model from the configuration data
            gptmodel = data["openai"]["model"]
            if not gptmodel:
                raise ValueError("Model is missing or empty in the configuration.")
        except KeyError:
            # Raise an error if the model is missing in the configuration
            raise KeyError("Model is missing in the configuration.")
        
        try:
            # Extract the organization from the configuration data, if available
            organization = data["openai"].get("organization", None)
        except KeyError:
            # Raise an error if the organization is missing in the configuration
            raise KeyError("Organization is missing in the configuration.")

        try:
            # Extract and validate the maximum number of tokens from the configuration data
            maxtokens = int(data["openai"]["maxtokens"])
        except KeyError:
            # Raise an error if the max tokens setting is missing in the configuration
            raise KeyError("Max tokens is mandatory and missing in the configuration.")
        except ValueError:
            # Raise an error if the max tokens setting is not an integer
            raise ValueError("Max tokens must be an integer.")

        try:
            # Extract and validate the temperature setting from the configuration data
            temperature = float(data["openai"]["temperature"])
            if not (0 <= temperature <= 1):
                raise ValueError("Temperature must be between 0 and 1.")
        except KeyError:
            # Raise an error if the temperature setting is missing in the configuration
            raise KeyError("Temperature is mandatory and missing in the configuration.")
        except ValueError:
            # Raise an error if the temperature setting is not a float between 0 and 1
            raise ValueError("Temperature must be a float between 0 and 1.")

        # Initialize GPTCommons instance with the extracted settings
        commons = GPTCommons(api_key=api_key, gptmodel=gptmodel, maxtokens=maxtokens, temperature=temperature, organization=organization)
        return commons

    def get_api_key(self) -> str:
        """
        Retrieves the API key for accessing the GPT model.

        Returns:
        str: The API key.
        """
        return self.api_key

    def get_gptmodel(self) -> str:
        """
        Retrieves the current GPT model configuration.

        Returns:
        str: The GPT model configuration.
        """
        return self.gptmodel

    def get_organization(self) -> str:
        """
        Retrieves the organization identifier.

        Returns:
        str: The organization identifier.
        """
        return self.organization

    def get_maxtokens(self) -> int:
        """
        Retrieves the maximum number of tokens allowed in a response.

        Returns:
        int: The maximum number of tokens.
        """
        return self.maxtokens

    def get_temperature(self) -> float:
        """
        Retrieves the temperature setting for the model, which controls the randomness of the output.

        Returns:
        float: The temperature setting.
        """
        return self.temperature

    def reduce_to_max_tokens(self, text) -> str:
        """
        Reduces the input text to a maximum number of tokens for the specified OpenAI model.

        Args:
        text (str): The input text to be reduced.

        Returns:
        str: The reduced text.
        """
        if not isinstance(self.get_maxtokens(), int):
            raise ValueError("max_tokens must be an integer.")

        # Initialize the tokenizer for the specified model
        tokenizer = tiktoken.encoding_for_model(self.get_gptmodel())

        # Tokenize the input text
        tokens = tokenizer.encode(text)

        # Truncate the tokens to the maximum allowed number
        truncated_tokens = tokens[:self.get_maxtokens()]

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

    def get_chat_completion(self, prompt) -> str:
        """
        Retrieves a completion using the OpenAI ChatCompletion API with the specified model and parameters.

        Args:
        prompt (str): The user's input or prompt for generating the completion.

        Returns:
        str: The generated completion.
        """
        response = self.client.chat.completions.create(
                messages=[
                        {
                            "role": "system",
                            "content": prompt,
                         }
                        ],
                model=self.get_gptmodel(),
                temperature=self.get_temperature()
                )
        return response.choices[0].message.content.strip()
    
    def get_arg(self, arg_name, arg_descriptions, default=None) -> str:
        """
        Retrieves the value of a command-line argument by its name from the sys.argv list.

        Args:
        arg_name (str): The name of the command-line argument to retrieve.
        arg_descriptions (dict): A dictionary containing argument names as keys and their descriptions as values.
        default: The default value to return if the argument is not found (default is None).

        Returns:
        str or default: The value of the specified command-line argument or the default value if not found.

        If '--help' is present in the command-line arguments, it prints the usage message along with descriptions of all arguments and exits the program.

        Example:
        >>> # Assuming the command-line arguments are ['--lang', 'English', '--url', 'example.com']
        >>> get_arg('--lang', {'--lang': 'Language (default: English)', '--url': 'PDF URL'}, 'Spanish')
        'English'
        >>> get_arg('--url', {'--lang': 'Language (default: English)', '--url': 'PDF URL'}, 'localhost')
        'example.com'
        >>> get_arg('--port', {'--lang': 'Language (default: English)', '--url': 'PDF URL'}, 8080)
        8080
        """
        if "--help" in sys.argv:
            script_name = sys.argv[0]
            print(f"Usage: python {script_name} [options]")
            print("Options:")
            for arg, desc in arg_descriptions.items():
                print(f"\t{arg}\t\t{desc}")
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
        # Clean the input text
        text = self.clean_text(text)

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