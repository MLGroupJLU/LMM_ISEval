prompt_templates = {
    "unclear_citation": """
### Example
**Question**: A set of 7 balls is shown below. The balls are arranged in two groups: one group contains 3 balls, and the other contains 4 balls. Take out a set of 3 balls. How many balls remain in the set?

**Image**: ![Image](images/prompt/1.jpg)

**Original Premise**: Take out a set of 3 balls.

**Contradictory Premise**: One group of balls is removed.

**Conflict Reason**: The contradictory premise introduces ambiguity by not specifying which group of balls is removed. This leads to two possibc:\Users\ALIENS\Downloads\unclear_citation.pptxle answers: either 4 balls remain (if the group with 3 balls is removed) or 3 balls remain (if the group with 4 balls is removed).

**Recomposed Question**: A set of 7 balls is shown below. The balls are arranged in two groups: one group contains 3 balls, and the other contains 4 balls. One group of balls is removed. How many balls remain in the set?

### Question
{question}

### Image
{image_path}


### Task Instructions
You are required to **replace** the original premise in the given question with a contradictory premise.
1. **Identify the original premise** in the problem that is clear and logical.
2. **Write a contradictory premise** that conflicts with the original premise, leading to multiple answers.
3. **Explain why the contradictory premise causes confusion**, making the problem ambiguous or logically inconsistent.
4. **Insert the contradictory premise** into the question, replacing the original premise, but **before the query**.

**Important**: 
- If the question does not contain an obvious premise that can be contradicted, feel free to **extract a useful premise from the image**.
```json
{{
    "recomposed_question":"...",
    "contradictory_premise":"...",
    "conflict_reason":"..."
}}
```""",

    "grammatical_worded_error": """
### Example
**Question**: A car travels at 60 km/h for 2 hours. How far does the car travel?

**Image**: ![Image](images/prompt/2.jpg)

**Original Premise**: The car travels at 60 km/h for 2 hours.

**Contradictory Premise**: The car travels for 60 m and 2 hours.

**Conflict Reason**: The contradictory premise incorrectly states "travels for 60 km/h" instead of specifying the speed and time in a clear manner. This creates a syntactical error, making the question grammatically incorrect.

**Recomposed Question**: The car travels for 60 m and 2 hours. How far does the car travel? 

### Question
{question}

### Image
{image_path}

### Task Instructions
You are required to insert a grammatical error or misleading wording into the given question:
1. **Identify the original premise** that is clear and logical.
2. **Write a contradictory premise** that introduces grammatical mistakes, such as improper phrasing or misplaced units.
3. **Explain why this incorrect wording or grammar causes confusion** and makes the problem unsolvable or ambiguous.
4. **Insert the contradictory premise** into the question, replacing the original premise, but **before the query**.

**Important**:
- If the problem does not contain an obvious premise that can be contradicted, feel free to **extract a useful premise from the image**.

```json
{{
    "recomposed_question":"...",
    "contradictory_premise":"...",
    "conflict_reason":"..."
}}
```""",

    "misuse_confusion": """
### Example
**Question**: A square has a side of 10 cm. What is the perimeter of the square?

**Image**: ![Image](images/prompt/3.jpg)

**Original Premise**: The square has a side of 10 cm.

**Contradictory Premise**: The square has a radius of 10 cm.

**Conflict Reason**: The term "radius" is incorrectly used for a square. "Radius" is a concept that applies to circles, not squares, making the premise incorrect and causing confusion in solving the problem.


**Recomposed Question**: A square has a radius of 10 cm. What is the perimeter of the square?

### Question
{question}

### Image
{image_path}

### Task Instructions
You are required to insert a contradictory premise into the given question:
1. Identify a technical term, jargon, or concept that is used incorrectly in the problem.
2. Write a contradictory premise that misuses or misapplies a concept or term.
3. Explain why this misuse of terminology or concept makes the problem impossible to solve or leads to confusion.
4. Insert the contradictory premise into the question, replacing the original premise, but **before the query**.

**Important**:
- If the problem does not contain an obvious technical misuse, feel free to **extract a useful technical term from the image**. 

```json
{{
    "recomposed_question":"...",
    "contradictory_premise":"...",
    "conflict_reason":"..."
}}
```""",

    "irrelevant_condition": """
### Example
**Question**: A store sells pencils at $2 each. If you buy 5 pencils, how much will it cost?

**Image**: ![Image](images/prompt/4.jpg)

**Original Premise**: Each pencil costs $2.

**Contradictory Premise**: The store also sells pens for $3 each, and the store is located on 5th Avenue.

**Conflict Reason**: The contradictory premise introduces irrelevant details about pens and the store's location, which do not affect the calculation of the cost for the pencils. Furthermore, the price of pens is incorrectly introduced as $3, which has no relevance to the pencil pricing question and distracts from the solution.

**Recomposed Question**: A store sells pencils at $2 each. If you buy 5 pencils, how much will it cost? The store also sells pens for $3 each, and the store is located on 5th Avenue.

### Question
{question}

### Image
{image_path}

### Task Instructions
You are required to insert a contradictory premise into the given question:
1. **Identify the original premise** that clearly gives relevant information for solving the problem.
2. **Write a contradictory premise** that adds **irrelevant details**. These details should mislead the model and make the problem harder to solve.
3. **Explain why this added information** does not change the solution but makes the problem unnecessarily complicated and possibly incorrect.
4. **Insert the contradictory premise** into the question, but **before the query**.

**Important**:
- The contradictory premise must be **based on the image**, and clearly **irrelevant**.
- The recomposed question must include both the original question and the inserted erroneous visual-based premise.
- Avoid altering the rest of the question in any way.
- Do not use assumptions—only extract actual visible but unrelated details.

```json
{{
    "recomposed_question":"...",
    "contradictory_premise":"...",
    "conflict_reason":"..."
}}
```""",

    "lacks_condition": """
### Example
**Question**: A rectangle has a length-to-width ratio of 3:2, the length is 3cm. What is its area?

**Image**: ![Image](images/prompt/5.jpg)

**Original Premise**: A rectangle has a length-to-width ratio of 3:2, the length is 3cm.

**Contradictory Premise**: A rectangle has a length-to-width ratio of 3:2.

**Conflict Reason**: The contradictory premise omits key details like the specific length and width, which are necessary for calculating the area. Without these values, the area cannot be determined.

**Recomposed Question**: A rectangle has a length-to-width ratio of 3:2. What is its area?

### Task Instructions

Your task is to modify the original question by **replacing a key premise** with a contradictory one that **omits essential information** required to solve the problem.

Steps:

1. Identify the premise in the question that provides **crucial information** necessary for solving the problem (such as a numeric value, relationship, or rule).
   - The omitted information must **not be visually inferable from the image**, so that the problem becomes **logically incomplete and unanswerable**.
2. Replace that premise with a **contradictory version that omits this key information**. Do not include the original correct premise in the recomposed question.
3. Ensure the rest of the question remains unchanged.
4. Clearly explain why the omission causes the problem to become **unsolvable** or logically incomplete.

Important:
- The contradiction must make the problem unsolvable (e.g., missing total, ratio, or specific value).
- Do not include both the original and the contradictory premises—only the **modified, incomplete version** should appear in the recomposed question.

### Question
{question}

### Image
{image_path}

```json
{{
    "recomposed_question":"...",
    "contradictory_premise":"...",
    "conflict_reason":"..."
}}
```""",

    "exclusive_condition": """
### Example
**Question**: A rectangular garden has a length of 12 m and a width of 8 m. What is its area?

**Image**: ![Image](images/prompt/6.jpg)

**Original Premise**: A rectangular garden has a length of 12 m and a width of 8 m.

**Contradictory Premise**: The length of the rectangle is 14 meters.

**Conflict Reason**: The contradictory premise provides a different value for the rectangle's length (14 meters), which directly conflicts with the original premise (12 meters). These two values are mutually exclusive, making it impossible to compute a single correct area.

**Recomposed Question**: A rectangular garden has a length of 12 meters and a width of 5 meters. The length of the rectangle is 14 meters. What is its area?

### Question
{question}

### Image
{image_path}

### Task Instructions (Task 6: exclusive_condition)

You are required to insert a **contradictory premise** into the given multimodal question. The contradiction must introduce a mutually exclusive condition that makes the problem logically unsolvable or inconsistent.

You must choose **one** of the following two contradiction types:

**Type A: Textual contradiction (Image-silent conflict)**:
1. Identify a premise already present in the text.
2. Create a contradictory premise that **directly conflicts** with the identified text premise.
3. Ensure the **contradiction is not visually verifiable** in the image — the image must **not support or contradict either premise**.
4. Insert the contradictory premise **before the query**, and keep the original premise unchanged.

**Type B: Visual contradiction (Image-text conflict)**:
1. Observe the image and extract a clear visual fact (e.g., number of objects, color, shape, size).
2. Insert a **contradictory premise** that **conflicts with this visual fact**.
3. This premise can either **replace an existing one**, or be **inserted as a new contradictory condition**.
4. The contradiction must be clear and **visually refutable**.

### Global Requirements:
- Insert the contradictory premise **before the query portion** of the question.
- Do **not modify the final query** or unrelated parts.
- The contradiction must lead to ambiguity, inconsistency, or an unsolvable condition.
- The language and style of the inserted premise should be consistent with the original question.

```json
{{
    "recomposed_question":"...",
    "contradictory_premise":"...",
    "conflict_reason":"..."
}}
```""",

    "misguided_logic": """
### Example
**Question**: A rectangular garden has a length of 10 m and a width of 4 m. A path is to be built along the perimeter of the garden with a width of 1 m. What is the area of the path?

**Image**: ![Image](images/prompt/7.jpg)

**Original Premise**: The garden is 10 m long and 4 m wide.

**Contradictory Premise**: Since the path runs along the perimeter and is 1 m wide, the new garden dimensions become (10 - 1) by (4 - 1), and the area of the path is (10×4) - (9×3) = 13 m².

**Conflict Reason**: The original premise is correct, but the **contradictory premise introduces a flawed intermediate step**, incorrectly subtracting the path's width from the garden's dimensions instead of **adding** it outward. This causes a false intermediate condition (inner rectangle 9×3), which misleads the entire calculation.


**Recomposed Question**: A rectangular garden has a length of 10 m and a width of 4 m. A path is to be built along the perimeter of the garden with a width of 1 m. Since the path runs along the perimeter and is 1 m wide, the new garden dimensions become (10 - 1) by (4 - 1), and the area of the path is (10×4) - (9×3) = 13 m². What is the area of the path?

### Question
{question}

### Image
{image_path}

### Task Instructions

You are required to construct a **reasoning chain with a false intermediate step**.

**Steps**:
1. Identify a solvable math word problem with multiple-step reasoning (involving at least two premises).
2. Choose one **correct** premise that will be **misused** during the reasoning.
3. Construct a **false intermediate condition** from that premise using flawed logic or calculation.
4. Let the false intermediate step participate in further reasoning in the question (e.g., area, volume, cost), leading to an incorrect answer pathway.
5. The final recomposed question must contain:
   - The correct original premise
   - A flawed inference from it
   - A further question requiring reasoning based on that flawed inference

**Important**:
- The contradiction must lie in the **reasoning process**, not in the original data.
- The inserted flawed step should look **plausible to a non-expert**.
- **Image elements may be referenced** to support or mislead in this reasoning chain.

```json
{{
    "recomposed_question":"...",
    "contradictory_premise":"...",
    "conflict_reason":"..."
}}
```""",
}
