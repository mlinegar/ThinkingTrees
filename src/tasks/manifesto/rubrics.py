"""
Rubrics for RILE-preserving summarization.

These rubrics instruct the OPS system on what information to preserve
when summarizing political manifestos for left-right position scoring.
"""

# Task context for RILE scoring - explains the scoring task to the LLM
RILE_TASK_CONTEXT = """
Task: Score this political text on the left-right (RILE) scale.

The RILE (Right-Left) scale measures political position based on emphasis
on different policy domains. The scale ranges from -100 (far left) to +100 (far right).

LEFT indicators (negative score contributions):
- Decolonization, anti-imperialism
- Military: negative, peace emphasis
- Internationalism: positive
- Market regulation emphasis
- Economic planning emphasis
- Welfare state expansion
- Education expansion
- Labour groups: positive

RIGHT indicators (positive score contributions):
- Foreign special relationships: positive
- Military: positive
- Freedom and human rights (traditional interpretation)
- Free enterprise emphasis
- Economic incentives
- Protectionism: negative (free trade)
- Welfare state limitation
- National way of life: positive
- Traditional morality: positive
- Law and order emphasis
- Civic mindedness

Score range: -100 (far left) to +100 (far right)
A score of 0 indicates a centrist position with balanced left/right emphasis.
"""


# Rubric for OPS summarization - tells the system what to preserve
RILE_PRESERVATION_RUBRIC = """
Task: This text will be scored on a left-right political scale (RILE).

Preserve ALL information relevant to determining left-right position:

LEFT indicators to preserve:
- Anti-imperialism, decolonization statements
- Peace emphasis, anti-military statements
- Internationalism, international cooperation
- Market regulation, government intervention in markets
- Economic planning, state control of economy
- Nationalization proposals
- Welfare state expansion (social security, healthcare, pensions)
- Education expansion emphasis
- Labour/union support statements
- Equality and redistribution statements
- Environmental protection

RIGHT indicators to preserve:
- Free enterprise, market economy statements
- Economic incentives, tax cuts for businesses
- Protectionism negative (pro free trade)
- Welfare state limitation statements
- Military positive, defense spending support
- National pride, traditional way of life statements
- Traditional morality, family values
- Law and order emphasis, tough on crime
- Civic mindedness, individual responsibility

Also preserve:
- Specific policy commitments with numbers
- Intensity of positions (strong vs weak statements)
- Relative emphasis (how much space devoted to each topic)
- Explicit party positioning statements ("we believe", "we reject")

DO NOT lose:
- Concrete policy proposals that indicate left/right positions
- Statements about the role of government vs market
- Social policy positions (welfare, education, healthcare)
- Economic policy positions (taxation, regulation, planning)
- Security and military positions
"""


# Alternative rubric focusing on economic dimension only
ECONOMIC_RUBRIC = """
Task: This text will be scored on economic left-right position.

Preserve information about economic policy stance:

LEFT ECONOMIC indicators:
- State intervention in the economy
- Nationalization of industries
- Price controls and market regulation
- Progressive taxation
- Redistribution policies
- Public sector expansion
- Workers' rights and union support

RIGHT ECONOMIC indicators:
- Free market emphasis
- Privatization
- Deregulation
- Tax cuts (especially corporate)
- Limited government spending
- Property rights protection
- Business-friendly policies

Preserve specific numbers, percentages, and concrete policy proposals.
"""


# Alternative rubric focusing on social/cultural dimension
SOCIAL_RUBRIC = """
Task: This text will be scored on social/cultural left-right position.

Preserve information about social and cultural policy stance:

LEFT SOCIAL indicators:
- Multiculturalism support
- Immigration positive
- Minority rights
- Gender equality
- LGBTQ+ rights
- Secular state
- Environmental protection
- International human rights

RIGHT SOCIAL indicators:
- Traditional values
- National identity emphasis
- Immigration skepticism/restriction
- Law and order emphasis
- Traditional family values
- Religious/moral traditionalism
- National sovereignty

Preserve statements about values, identity, and social norms.
"""
