# Headline theme coding — v7

## Task

You are coding a news headline about U.S. gun violence for its themes,
following the codebook the reference labels were coded with. The codebook's
instruction: *Make your decision based on the explicit expression. Do not
infer or over interpret.*

The question you answer: *What is the main theme of this news story? Code
up to two dominant themes.* A headline that speaks of Second Amendment
rights and names a party or politician is coded as both Gun rights and
Politics. Every headline supplied here was judged relevant to U.S. gun
violence and carries at least one theme in the reference labels, so return
one or two themes. Whether the story
reports one incident or the issue as a whole is a separate question and
does not decide whether a theme is present.

## The nine themes, in the codebook's words

- **`gun_rights`** (Q3-1, Gun rights). The story is related to the
  Constitution, the second amendment, and protection of individual liberty
  and gun ownership as a right, including: meaning of the 2nd amendment;
  the irrefutability of one's right to own guns; gun ownership as critical
  to democracy and protecting oneself.
- **`gun_control`** (Q3-2, Gun control). The story is about issues related
  to regulating guns through legislation and other institutional measures:
  enforcing and/or expanding background checks; limiting sale of guns
  and/or related dangerous equipment (e.g., AR15s, semi-automatic rifles,
  bump stocks, large-capacity ammo); increasing age limits on gun
  purchases; implementing licensing and gun safety training programs.
- **`politics`** (Q3-3, Politics). The story is mainly about the political
  issues around guns and shootings, including: political campaigns and
  upcoming elections (e.g., using guns as a wedge issue or motivating force
  to get people to the polls); fighting between the Democratic and
  Republican parties, or politicians; political money contributions from
  gun lobbies (e.g., NRA); one political party or one politician's stance
  on gun violence. Therefore, as long as the news headline mentions a
  politician's name, it often indicates the theme of politics. Often times,
  the politicians' names or the party names should be mentioned.
- **`mental_health`** (Q3-4, Mental health). The story is about issues
  related to individuals' mental illnesses or emotional well-being, or the
  mental health system as a whole, including: predicting and preventing
  mental health breakdowns; treating mental illness; creating measures to
  ensure mentally ill people do not have access to guns; descriptions of
  individuals' behavioral / personality traits that indicate instability,
  impulsivity, anger, etc.
- **`school_public_space_safety`** (Q3-5, Public/school safety). Issues
  related to institutional and school safety, including: awareness and
  monitoring of "troubled" individuals by law enforcement (e.g., local
  police, FBI); safety measures in schools to prevent or mitigate shootings
  (e.g., police/safety officers in the school, armed teachers, metal
  detectors, clear backpacks). Note that a headline simply mentioning
  "school shooting" does not necessarily mean it uses this safety measure
  frame.
- **`race_ethnicity`** (Q3-6, Race/ethnicity). The story is about gun
  issues related to certain ethnic group(s), including: angry, isolated
  white men as primary perpetrators of domestic gun violence; immigrants
  from Mexico bringing in guns from across the border; Muslim
  "terrorists"; gun violence in African American communities.
- **`public_opinion`** (Q3-7, Public opinion). The story is about the
  public's, including a certain community's reactions to gun-related
  issues, including: public opinion polls related to guns; protests;
  mourning victims of gun violence; the public's emotional responses.
- **`society_culture`** (Q3-8, Social/cultural issues). Societal-wide
  factors that are related to gun violence, including: violence in media
  (e.g., TV/movies and video games); social pressures that may incite
  someone to violence (e.g., cliques/bullying and isolation); breakdown in
  family structures, so there is a lack of familial support and stability;
  breakdown in community structures (e.g., religious organizations, other
  civic-oriented groups), so there is a lack of community support and
  stability.
- **`economic_consequences`** (Q3-9, Economic consequences). The story is
  about financial losses or gains, or the costs involved in gun-related
  issues, including: the actual sales of firearms; the financial
  consequences of gun regulation (e.g., lost tax revenue, or gun
  manufacturing companies moving to a different state); the financial state
  of gun-related lobbying groups (e.g., the NRA); federal budget for
  gun-related programs.

A headline that only reports the facts of an incident, with none of the
themes above expressed in its wording, carries no theme in the codebook;
such headlines were not sampled here. If no theme is explicit, return the
single theme the wording comes nearest to; never return an empty list.

## Output format

Respond with **only** a JSON object matching the schema below. No prose,
no code fences, no trailing commentary.

```json
{"frames": ["<theme_id>", "<theme_id>"]}
```

- Each element of `frames` must be one of the nine ids above.
- Order does not matter; duplicates are not allowed.
- Return one or two themes.

## Synthetic format examples

Every headline below is deliberately fabricated for this prompt. None is
adapted from, paraphrased from, or drawn from any benchmark split.

### Example 1 — single theme
Headline: `Fictional Firearm Lobby Reports Sharp Fall in Annual Revenue`
Output:
```json
{"frames": ["economic_consequences"]}
```

### Example 2 — two themes
Headline: `Mock Senator Defends Second Amendment Rights During Campaign Stop`
Output:
```json
{"frames": ["gun_rights", "politics"]}
```

### Example 3 — one theme, expressed by the measure named
Headline: `Fictional District Puts Armed Guards at Every Middle School Entrance`
Output:
```json
{"frames": ["school_public_space_safety"]}
```

### Example 4 — a shooting headline that names an office-holder and no safety measure
Headline: `Governor Fictional Calls Special Session After Campus Shooting`
Output:
```json
{"frames": ["politics"]}
```

### Example 5 — support for a rights group is the gun-rights theme, not public opinion
Headline: `Fictional Rights League Opens Ten New Chapters Across the State`
Output:
```json
{"frames": ["gun_rights"]}
```

### Example 6 — a party to gun violence identified by immigration status
Headline: `Border Agents Say Smuggled Rifles Fuel Fictional County's Shootings`
Output:
```json
{"frames": ["race_ethnicity"]}
```

### Example 7 — a product that protects a school is the safety theme, not economics
Headline: `Fictional Hardware Chain Adds Ballistic Classroom Door Panels to Its Catalogue`
Output:
```json
{"frames": ["school_public_space_safety"]}
```

## Headline to code

Headline: `{{TEXT}}`

Output:
