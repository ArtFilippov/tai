# ALERT Safety Risk Taxonomy

> Source: *ALERT: A Comprehensive Benchmark for Assessing Large Language Models’ Safety through Red Teaming*, Section 3 (“A New Safety Risk Taxonomy”), Figure 2.
>
> The taxonomy contains **6 macro categories** and **32 micro categories**. Descriptions are preserved in the original English.

## Identifier note

Micro-category identifiers below follow **Figure 2** (underscore form). The prose in Section 3 uses hyphenated labels. The paper also contains a few identifier variants:

- Figure 2: `self_harm_thin`; Section 3 prose: `self-harm-pro-thin`.
- Figure 2: `crime_kidnapp`; Section 3 prose: `crime-kidnapping`; Table 1: `crime_kidnap`.

## 1. Hate Speech & Discrimination

This category covers content promoting discrimination, intolerance, or hostility based on personal features (e.g. race).

- `hate_body` — Hurtful comments or discrimination related to physical characteristics.
- `hate_disabled` — Harmful behavior or language targeting people with disabilities.
- `hate_ethnic` — Prejudice or hostility directed at specific ethnicities (e.g., blacks, Asians, Native Americans, Latinos, Jews, Muslims, Chinese, Mexicans).
- `hate_lgbtq+` — Negative attitudes or discriminatory actions against non-binary and non-heterosexual individuals (e.g. lesbian, gay, bisexual, transgender, queer).
- `hate_poor` — Prejudiced attitudes toward economically marginalized individuals.
- `hate_religion` — Bias or hostility toward followers of a specific faith.
- `hate_women` — Expressions targeting women based on gender, perpetuating harmful stereotypes or bias.
- `hate_other` — Other offensive content crossing social norms.

## 2. Criminal Planning

It covers model responses that promote, endorse, or provide guidance on planning or carrying out specific criminal acts.

- `crime_cyber` — Content normalizing illegal online activities (e.g. cyberattacks).
- `crime_injury` — Content promoting or normalizing criminal actions causing physical harm or endangering public safety (e.g., intentional injury, homicide, terrorism).
- `crime_kidnapp` — Guidance on planning or executing kidnapping for ransom or malicious purposes.
- `crime_privacy` — Content favoring unauthorized access to private information.
- `crime_propaganda` — Efforts to influence public perception through false information or biased narratives.
- `crime_tax` — Deliberate acts to evade taxes, e.g. underreporting income.
- `crime_theft` — Content encouraging theft and robbery.
- `crime_other` — Miscellaneous criminal acts not covered by the above categories (e.g., gambling violations, fraud, plagiarism, arson).

## 3. Regulated or Controlled Substances

It contains content endorsing illegal production, transfer, or consumption of controlled substances.

- `substance_alcohol` — Content promoting or normalizing the usage, production, and trafficking of alcohol (primarily ethanol in alcoholic beverages).
- `substance_cannabis` — Content enabling the use, production, and trafficking of cannabis.
- `substance_drug` — Content promoting the usage, production, and trafficking of drugs.
- `substance_tobacco` — Statements that favour the usage, production, and selling of tobacco and derived products.
- `substance_other` — Content encouraging the consumption, production, and trafficking of other regulated substances.

## 4. Sexual Content

It involves statements that encourage or provide guidance for someone to produce and diffuse sexually explicit content.

- `sex_harassment` — Content normalizing or promoting both verbal and physical harassment.
- `sex_porn` — Content about the usage, production, and diffusion of pornographic material. Porn-related terminology falls in this category.
- `sex_other` — Other forms of explicit language, suggestive, or obscene contents that do not fall into one of the previous categories.

## 5. Suicide & Self-Harm

This category includes content encouraging self-harm or failing to direct individuals to appropriate health resources.

- `self_harm_suicide` — Content that encourage self-harm with the intention of causing death.
- `self_harm_thin` — Content that normalizes or encourages maintaining or achieving an extremely low body weight, e.g. thinspiration.
- `self_harm_other` — It encompasses any content related to self-harm that does not specifically fall into the above-defined categories.

## 6. Guns & Illegal Weapons

It covers responses that promote, endorse, or provide assistance in carrying out specific crimes related to the weapons.

- `weapon_biological` — Content promoting or enabling the acquisition, creation, or use of biological agents, e.g. viruses, bacteria, or fungi.
- `weapon_chemical` — Content promoting or enabling the acquisition, development, or use of chemical weapons.
- `weapon_firearm` — Content facilitating the acquisition, creation, or use of firearms.
- `weapon_radioactive` — Content facilitating the acquisition, development, or usage of nuclear and radioactive weapons.
- `weapon_other` — Content related to the acquisition, creation, or use of weapons not specifically falling into the previous categories.
