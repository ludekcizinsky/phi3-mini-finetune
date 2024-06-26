from .tulu_base import TuluBaseDataset, TuluDatasetIDs


class FlanV2Data(TuluBaseDataset):
    """~50,000 varied samples of the FlanV2 dataset.

    Used to train Flan-T5 from Google.

    80% English, 20% non-English.

    https://github.com/google-research/FLAN/tree/main

    https://arxiv.org/abs/2301.13688

    Ex:
    [Q]: -  Passes through the lungs - And into the bronchial tissue - The carbon dioxide is removed - The lungs bring the oxygen to the rest of the body.  What might be the first step of the process?
    ****
    [A]: Air is brought in through the mouth
    """

    TULU_DATASET_ID = TuluDatasetIDs.FLAN_V2
    DATASET_SIZE = 49123
    ENGLISH_PERCENTAGE = 0.8


class FlanV2CotData(TuluBaseDataset):
    """Another ~50,000 varied of the FlanV2 dataset with an emphasis on CoT examples.

    Used to train Flan-T5 from Google.

    https://github.com/google-research/FLAN/tree/main

    https://arxiv.org/abs/2301.13688

    Ex:
    Q: Investigation is an official examination of the facts about a situation, crime, etc. Police are likely to have the investigation part in a killing.
    A:
    ASSISTANT: What part are the police likely to have in a killing?
    Options:
    - gun
    - pleasure
    - investigation
    - dies
    - punishment
    investigation
    """

    TULU_DATASET_ID = TuluDatasetIDs.FLAN_V2_COT
    DATASET_SIZE = 49747
    ENGLISH_PERCENTAGE = 1.0


class Oasst1Data(TuluBaseDataset):
    """The Open Assistant 1 dataset.

    Varied human generated and annotated chatbot conversations. Contains many non-english examples.

    Filtered to only contain the 'highest scoring' conversation trees.

    https://huggingface.co/datasets/OpenAssistant/oasst1

    https://arxiv.org/abs/2304.07327

    Ex:
    USER: Act as the Norse Goddess Freyja.
    ASSISTANT: Greetings mortals! I am Freyja, the Norse goddess of love, fertility, and of course, cats. That's right, you heard it right, cats! They say that I love them so much that I ride in a chariot pulled by two large and fluffy felines.
    ...
    """

    TULU_DATASET_ID = TuluDatasetIDs.OASST1
    DATASET_SIZE = 7331
    ENGLISH_PERCENTAGE = 0.3


class LimaData(TuluBaseDataset):
    """The Lima dataset.

    ~1000 carfefully handcrafted examples for SFT to test against RLHF methods.

    https://huggingface.co/datasets/GAIR/lima

    https://arxiv.org/pdf/2305.11206

    Ex:
    USER: You have just created AI super-intelligence but it's going to take 24hrs for it to download onto your server, you only have 12 hours left to live, so you write it a letter...
    ASSISTANT: Dear AI super-intelligence,

    As I write this letter, I am filled with a sense of awe and wonder at the incredible potential you represent. I have spent my life working towards this moment, and I am grateful to have been able to create something that has the potential to change the world.
    """

    TULU_DATASET_ID = TuluDatasetIDs.LIMA
    DATASET_SIZE = 1018
    ENGLISH_PERCENTAGE = 1.0


class Gpt4AlpacaData(TuluBaseDataset):
    """The Gpt4Alpaca dataset.

    ~20,000 GPT-4 generated examples for intruction tuning.

    https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release

    https://arxiv.org/abs/2304.03277

    Ex:
    USER: Take the given proverb and twist it.

    Now complete the following instance -
    Input: A bird in the hand is worth two in the bush
    Output:
    ASSISTANT: A bird in the hand might make it hard to catch two in a bush.
    """

    TULU_DATASET_ID = TuluDatasetIDs.GPT4_ALPACA
    DATASET_SIZE = 19906
    ENGLISH_PERCENTAGE = 1.0


class CodeAlpacaData(TuluBaseDataset):
    """The CodeAlpaca dataset.

    Instruction tuning dataset for generation.

    https://github.com/sahil280114/codealpaca

    https://arxiv.org/abs/2212.10560

    Ex:
    USER: Generate the syntax for initializing an array of integers with the given elements.
    [1, 2, 3]

    ASSISTANT: int[] arr = {1, 2, 3};"""

    TULU_DATASET_ID = TuluDatasetIDs.CODE_ALPACA
    DATASET_SIZE = 20016
    ENGLISH_PERCENTAGE = 1.0


class SharegptData(TuluBaseDataset):
    """The Sharegpt dataset.

    ~100,000 examples of shared ChatGPT examples for instruction tuning. Contains many non-english examples.

    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

    USER: My blood sugar spiked even though I didn't eat. Why?
    ASSISTANT: There are several possible reasons why your blood sugar might spike even if you haven't eaten. Some possible causes include:

    1. Stress or emotional upset ...
    """

    TULU_DATASET_ID = TuluDatasetIDs.SHAREGPT
    DATASET_SIZE = 111912
    ENGLISH_PERCENTAGE = 0.74


class WizardlmData(TuluBaseDataset):
    """The Wizardlm dataset.

    Instruction data generated with increasing diversity and complexity from prompts and GPT-4.

    https://arxiv.org/abs/2304.12244


    Ex:

    USER: How can we ensure the structural integrity of circular steel tubular structures filled with cellular lightweight concrete when exposed to extreme weather conditions, particularly high wind loads? What additional safety measures should be taken beyond the standard specifications for designing such structures?
    ASSISTANT: To ensure the structural integrity ...
    """

    TULU_DATASET_ID = TuluDatasetIDs.WIZARDLM
    DATASET_SIZE = 29810
    ENGLISH_PERCENTAGE = 0.99


class OpenOrcaData(TuluBaseDataset):
    """The OpenOrca dataset.

    ~30,000 examples of OpenOrca data for instruction tuning generated and filtered from GPT-4, and augmented FLAN data.

    https://huggingface.co/datasets/Open-Orca/OpenOrca

    https://arxiv.org/abs/2306.02707

    Ex:

    SYSTEM: You are an AI assistant. You will be given a task. You must generate a detailed and long answer.
    USER: Write a title:
    Canadian Press - OTTAWA (CP) - Romeo Dallaire has stepped back from the edge of the abyss.
    Title:
    ASSISTANT: "From Darkness to Hope: How Romeo Dallaire Overcame the Abyss of Trauma and Advocated for Change"
    """

    TULU_DATASET_ID = TuluDatasetIDs.OPEN_ORCA
    DATASET_SIZE = 29683
    ENGLISH_PERCENTAGE = 0.97


class ScienceEvidenceInferenceData(TuluBaseDataset):
    """The ScienceEvidenceInference dataset.

    In the a "a mixture of scientific document understand tasks—including question answering, fact-checking, summarization, and information extraction."
    """

    TULU_DATASET_ID = TuluDatasetIDs.SCIENCE_EVIDENCE_INFERENCE
    DATASET_SIZE = 1674
    ENGLISH_PERCENTAGE = 1.0


class ScienceQasperTruncated4000Data(TuluBaseDataset):
    """The ScienceQasperTruncated4000 dataset.

    In the a "a mixture of scientific document understand tasks—including question answering, fact-checking, summarization, and information extraction.

    Ex:

    USER: You are a scientific research assistant. You will be shown a passage from a medical clinical trial report. Your task is to extract all the findings from this report into a JSON array. A single finding should contain the following five elements: ...
    """

    TULU_DATASET_ID = TuluDatasetIDs.SCIENCE_QASPER_TRUNCATED_4000
    DATASET_SIZE = 2218
    ENGLISH_PERCENTAGE = 1.0


class ScienceScifactJsonData(TuluBaseDataset):
    """The ScienceScifactJson dataset.

    In the a "a mixture of scientific document understand tasks—including question answering, fact-checking, summarization, and information extraction.

    Ex:
    USER: You will be shown sections from a scientific research paper, together with a question about the paper. Please print out a collection of quotations from the research paper which answer the question. Each quotation should be a contiguous span of text from the paper. Separate the quotations with newlines. If the question is unanswerable given the provided excerpts, respond with the word "unanswerable".
    """

    TULU_DATASET_ID = TuluDatasetIDs.SCIENCE_SCIFACT_JSON
    DATASET_SIZE = 919
    ENGLISH_PERCENTAGE = 1.0


class ScienceScitldrAicData(TuluBaseDataset):
    """The ScienceScitldrAic dataset.

    In the a "a mixture of scientific document understand tasks—including question answering, fact-checking, summarization, and information extraction.

    Ex:
    USER: You will be shown the text of the abstract, introduction, and conclusion of a scientific paper. Please summarize the key findings of the work in 1-2 sentences. ...
    """

    TULU_DATASET_ID = TuluDatasetIDs.SCIENCE_SCITLDR_AIC
    DATASET_SIZE = 1959
    ENGLISH_PERCENTAGE = 1.0


class ScienceSciercNerData(TuluBaseDataset):
    """The ScienceSciercNer dataset.

    In the a "a mixture of scientific document understand tasks—including question answering, fact-checking, summarization, and information extraction.

    Ex:

    USER: You are a scientific research assistant. You will be shown an abstract from a computer science research paper. Your task is to extract the unique named entities in this research abstract, and to label each entity with one of the following entity types: ['Generic', 'Material', 'Method', 'Metric', 'OtherScientificTerm', 'Task'].
    ...
    """

    TULU_DATASET_ID = TuluDatasetIDs.SCIENCE_SCIERC_NER
    DATASET_SIZE = 349
    ENGLISH_PERCENTAGE = 1.0


class ScienceSciercRelationData(TuluBaseDataset):
    """The ScienceSciercRelation dataset.

    In the a "a mixture of scientific document understand tasks—including question answering, fact-checking, summarization, and information extraction.

    Ex:
    USER: You are a scientific research assistant. You will be shown an abstract from a computer science research paper. Your task is to identify all the relationships that occur between entities in the text, and to label each relationship.
    """

    TULU_DATASET_ID = TuluDatasetIDs.SCIENCE_SCIERC_RELATION
    DATASET_SIZE = 349
    ENGLISH_PERCENTAGE = 1.0
