import re


LETTERS = list("ABCDEFGHIJ")

OFFICIAL_FIVE_SHOT_PROMPT = r"""Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

Question: 
A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is
A) 10
B) 40
C) 6
D) 25
E) 15
F) 50
G) 30
H) 4
I) 5
J) 20

Answer: Let's think step by step. In a refracting telescope, if both lenses are converging, the focus of both lenses must be between the two lenses, and thus the focal lengths of the two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the focal length of the other must be 80 cm. The magnification is the ratio of these two focal lengths, or 4.
Answer: H.

Question: 
Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye? 
A) 1000 times more
B) 50 times more
C) 5000 times more
D) 500 times more
E) 10000 times more
F) 20000 times more
G) 2000 times more
H) 100 times more
I) 10 times more
J) N/A

Answer: Let's think step by step. The amount of light a telescope can gather compared to the human eye is proportional to the area of its apertures. The area of a circle is given by the formula $A = \pi \left(\frac{{D}}{{2}}\right)^2$, where $D$ is the diameter. Therefore, the relative light-gathering power is calculated as:
\[
\frac{{\left(\frac{{50 \text{{ cm}}}}{{2}}\right)^2}}{{\left(\frac{{5 \text{{ mm}}}}{{2}}\right)^2}} = \frac{{\left(\frac{{50 \text{{ cm}}}}{{0.1 \text{{ cm}}}}\right)^2}}{{\left(\frac{{5 \text{{ mm}}}}{{0.1 \text{{ cm}}}}\right)^2}} = \frac{{500^2}}{{5^2}} = 10000.
\]
Answer: E.

Question: 
Where do most short-period comets come from and how do we know? 
A) The Kuiper belt; short period comets tend to be in the plane of the solar system like the Kuiper belt. 
B) The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt. 
C) The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt. 
D) The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud. 
E) The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud. 
F) The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud. 
G) The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt. 
Answer: Let's think step by step. Most short-period comets originate from the Kuiper belt. This is deduced from the observation that these comets tend to follow orbits that lie in the plane of the solar system, similar to the distribution of objects in the Kuiper belt itself. Thus, the alignment of these cometary orbits with the ecliptic plane points to their Kuiper belt origin.
Answer: A.

Question: 
Colors in a soap bubble result from light 
A) dispersion 
B) deflection 
C) refraction 
D) reflection 
E) interference 
F) converted to a different frequency 
G) polarization 
H) absorption 
I) diffraction 
J) transmission 

Answer: Let's think step by step. The colorful patterns observed in a soap bubble are caused by the phenomenon of light interference. This occurs when light waves bounce between the two surfaces of the soap film, combining constructively or destructively based on their phase differences and the varying thickness of the film. These interactions result in vibrant color patterns due to variations in the intensity of different wavelengths of light.
Answer: E.

Question: 
A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven? 
A) 240 W
B) 120 W
C) 10 W
D) 480 W
E) 360 W
F) 200 W
G) 30 W
H) 150 W
I) 60 W
J) 300 W

Answer: Let's think step by step. The rate of energy usage, known as power, in an electrical circuit is calculated by the product of voltage and current. For a microwave oven connected to a 120 V outlet and drawing a current of 2 amps, the power consumption can be calculated as follows:
\[
\text{{Power}} = \text{{Voltage}} \times \text{{Current}} = 120 \, \text{{V}} \times 2 \, \text{{A}} = 240 \, \text{{W}}.
\]
Therefore, the microwave oven uses energy at a rate of 240 watts.
Answer: A.

Question: 
{}

Answer: Let's think step by step. """


def doc_to_target(doc):
    return LETTERS.index(doc["answer_letter"])


def process_docs_cot(dataset):
    def _process_doc(doc):
        return {
            "choices": list(doc["options"]),
            "cot_target": f"({doc['answer_letter']})",
        }

    return dataset.map(_process_doc)


def process_docs_official_cot(dataset):
    def _process_doc(doc):
        return {
            "choices": list(doc["options"]),
            "answer_letter": doc["answer_letter"],
        }

    return dataset.map(_process_doc)


def doc_to_cot_text(doc):
    options = "\n".join(
        f"({LETTERS[idx]}) {option}" for idx, option in enumerate(doc["choices"])
    )
    return (
        f"The following is a question about {doc['field']}.\n"
        "Reason step by step, then finish with `The answer is (X)`, "
        "where X is the correct option letter.\n"
        f"Question:\n{doc['question']}\n"
        f"Choices:\n{options}\n"
        "Let's think step by step:"
    )


def doc_to_official_cot_text(doc):
    question_with_options = (
        doc["question"]
        + "\n"
        + "\n".join(
            f"{LETTERS[idx]}) {option}" for idx, option in enumerate(doc["choices"])
        )
    )
    return OFFICIAL_FIVE_SHOT_PROMPT.format(question_with_options)


def _safe_search(pattern, text, flags=0):
    try:
        return re.search(pattern, text, flags)
    except re.error:
        return None


def extract_official_option_label(text, options="ABCDEFGHIJ"):
    if not isinstance(text, str) or not isinstance(options, str):
        return "error"

    option_str = "".join(LETTERS[: len(options)]) if options else "ABCDEFGHIJ"
    text = text.rstrip()
    last_line = text.split("\n")[-1]
    boxed_prefix = r"(?:[\*\$\\{\(\[]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text)\{)?)*\s*"
    boxed_suffix = r"(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)"
    option_group = f"([{option_str}])"
    patterns = [
        r"\bthe\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*"
        + boxed_prefix
        + option_group
        + boxed_suffix,
        r"(?i:answer)[\*\s]*:\s*" + boxed_prefix + option_group + boxed_suffix,
        r"^[^\w\r\n]*" + boxed_prefix + option_group + boxed_suffix,
    ]

    for search_text in (last_line, text):
        for pattern in patterns:
            match = _safe_search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    return None


def extract_official_option_content(text, options_content):
    if not isinstance(text, str) or not isinstance(options_content, list):
        return "error"

    text = text.rstrip()
    last_line = text.split("\n")[-1]
    for search_text in (last_line, text):
        for idx, option_content in enumerate(options_content):
            if not isinstance(option_content, str) or not option_content:
                continue
            escaped_option = re.escape(option_content)
            patterns = [
                r"\bthe\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*"
                + escaped_option
                + r"(?:[\s:\.\*)]|$)",
                r"(?i:answer)\s*:?\s*" + escaped_option + r"(?:[\s:\.\*)]|$)",
                r"^[^\w\r\n]*" + escaped_option + r"(?:[\s:\.\*)]|$)",
            ]
            for pattern in patterns:
                if _safe_search(pattern, search_text, re.IGNORECASE):
                    return LETTERS[idx]
    return None


def extract_official_answer(response, choices):
    response_before_next_question = response.split("Question:")[0]
    prediction = extract_official_option_label(
        response_before_next_question, "ABCDEFGHIJ"
    )
    if prediction is None:
        prediction = extract_official_option_content(
            response_before_next_question, choices
        )
    if prediction is None:
        prediction = extract_official_option_label(response, "ABCDEFGHIJ")
    if prediction is None:
        prediction = extract_official_option_content(response, choices)
    return prediction


def process_results_official_cot(doc, results):
    prediction = extract_official_answer(results[0], doc["choices"])
    return {"exact_match": int(prediction == doc["answer_letter"])}
