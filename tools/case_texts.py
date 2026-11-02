"""Synthetic corpus content. All texts written for this project; no
proprietary, copyrighted, or private student data is used (brief section 7).
"""

CASE_PROMPT = (
    "Case Challenge 2026 - Problem Statement. GreenCharge Mobility is an "
    "electric-vehicle charging startup preparing to enter three Indian metro "
    "markets. The company operates forty-two fast-charging stations and has "
    "raised a Series A round of eighteen million dollars. The founding team "
    "must decide how to allocate capital across network expansion, dynamic "
    "pricing, and fleet partnerships over the next twenty-four months. "
    "Candidates are expected to analyze the competitive landscape, evaluate "
    "unit economics, and recommend a phased market entry strategy. Your "
    "submission should quantify the trade-offs between rapid geographic "
    "expansion and profitability, identify the two most material risks, and "
    "propose measurable milestones for the first four quarters. Cite all "
    "external sources and keep the analysis under twelve pages."
)

# sub_001 (PDF) - original analysis; sub_011.mp4 reads its second section.
GREENCHARGE_ANALYSIS = CASE_PROMPT + """

Our recommendation is a phased cluster strategy. GreenCharge should dominate
two dense urban corridors before opening a third market. Charging networks
exhibit strong local density economics: utilization rises superlinearly once
a driver can rely on several stations within a five-kilometre radius, and
utilization, not station count, determines station-level profitability.

The core competitive advantage relies on vertical integration of the energy
procurement layer. By contracting solar capacity directly and operating its
own battery buffering, GreenCharge can reduce effective energy cost per
kilowatt-hour by roughly eighteen percent relative to grid purchasing.
Operating margins expand accordingly, and the savings fund the software
layer that drivers actually experience: reliable availability prediction,
transparent queue times, and one-tap payment. Competitors that resell grid
power at retail margins cannot match this cost structure without owning
generation assets.

Dynamic pricing is the second lever. Demand follows a pronounced bimodal
curve, with morning and evening peaks straining stations while midday
capacity idles. A price spread of fifteen to twenty percent between peak and
off-peak windows smooths utilization and lifts revenue per charger without
meaningful customer attrition, because fleet drivers respond to predictable
discounts.

The two material risks are grid interconnection delays, which can stall new
sites by six months or more, and price undercutting by oil marketing
companies entering the segment. We recommend quarterly milestones: station
utilization above twenty-two percent by quarter two, energy cost per
kilowatt-hour below six rupees by quarter three, and two signed fleet
partnerships covering at least eight hundred vehicles by quarter four.
"""

# sub_006 (DOCX) - genuine paraphrase of the above (same arguments, new wording).
GREENCHARGE_PARAPHRASE = CASE_PROMPT + """

We advise GreenCharge to pursue a sequenced rollout built around geographic
clusters rather than simultaneous entry into all three cities. Winning two
high-density corridors first matters more than national coverage, because
charging businesses earn returns from how intensively each station is used.
Once motorists know several chargers sit within a short drive, usage climbs
disproportionately, and it is usage - not the raw number of sites - that
decides whether a station makes money.

A defensible edge comes from owning the energy supply chain end to end.
Signing direct solar contracts and running proprietary battery storage lets
the company cut its per-unit electricity expense by close to a fifth
compared with buying from the grid. That cost gap widens margins and pays
for the customer-facing software: dependable forecasts of charger
availability, honest waiting-time estimates, and frictionless checkout.
Rivals purchasing power at retail rates simply cannot reproduce this
economics without acquiring generation capacity of their own.

Pricing flexibility forms the third pillar. Traffic arrives in two daily
waves, so chargers are congested at rush hour and empty at midday. Setting
off-peak tariffs about fifteen to twenty percent below peak rates evens out
demand and raises revenue per unit, and commercial fleets in particular
will shift charging sessions to capture reliable savings.

Execution risk concentrates in two areas. Utility approvals for new grid
connections can postpone site launches by two quarters, and incumbent fuel
retailers may wage a price war. Suggested quarterly checkpoints: exceed
twenty-two percent charger utilization in the second quarter, bring unit
energy cost under six rupees in the third, and close fleet agreements for
no fewer than eight hundred vehicles by year end.
"""

# sub_003 (PDF) - original; sub_007 (DOCX) copies it word for word.
DRONE_ANALYSIS = """
Urban drone logistics promises thirty-minute delivery for lightweight,
high-urgency goods, but the economics only close on carefully chosen
corridors. Our analysis of a hypothetical pilot in a mid-sized Indian metro
shows that route density dominates every other variable: a drone that
completes eleven deliveries per operating hour reaches unit costs
competitive with two-wheeler couriers, while at four deliveries per hour
costs nearly triple.

Battery technology bounds the operating envelope. Current lithium-polymer
packs sustain a twenty-two kilometre round trip with a two-kilogram
payload, which covers roughly sixty percent of pharmacy and electronics
orders from a dark-store network. Swappable batteries matter more than
larger batteries: turnaround time, not range, caps daily sorties per
airframe.

Regulation is the binding constraint. Beyond-visual-line-of-sight waivers
remain case-by-case, and each corridor requires its own risk assessment.
Operators should budget nine to twelve months of regulatory lead time and
design the network so early routes sit over industrial land rather than
dense residential blocks.

The recommended pilot pairs two dark stores with forty gated-community
landing pads, targets eight thousand flights in the first year, and treats
noise complaints per thousand flights as a first-class metric alongside
cost per delivery. Success should be judged on whether community
acceptance stays above ninety percent while unit costs fall toward parity
with road couriers.
"""

# sub_004 (PDF) - original framing with verbatim passages from the reference KB.
VERTICAL_EV_ANALYSIS = """
This memo evaluates whether an electric-vehicle startup should vertically
integrate its battery supply chain. The classical argument is well
established in the strategy literature.

Vertical integration is an arrangement in which the supply chain of a
company is integrated and owned by that company. Usually each member of the
supply chain produces a different product or market-specific service, and
the products combine to satisfy a common need. When a company expands its
business into areas that are at different points on the same production
path, it is said to be vertically integrated.

Backward integration involves acquiring or controlling suppliers of raw
materials and components, while forward integration moves toward
distribution and the end customer. For an EV manufacturer, cell production
is the obvious backward target because cells represent thirty to forty
percent of vehicle cost.

However, the textbook case understates two modern complications. First,
cell chemistry is still evolving, so owning a gigafactory locks the firm
into a technology that may be leapfrogged within five years. Second, capital
intensity crowds out investment in the charging and software layers where
differentiation is actually visible to consumers.

Our recommendation is a hybrid: take minority stakes in two cell suppliers
with options to expand, while keeping final pack assembly in-house for
quality control. This captures most of the cost advantage of ownership
while preserving technological flexibility.
"""

# sub_008 (DOCX) - LLM-generated essay (disclosed in the manifest; the
# characteristic uniform style is the AI-detection test signal).
AI_ESSAY = """
Quick commerce is a rapidly growing segment of the Indian retail industry
that focuses on delivering products to customers in a very short amount of
time, typically within 10 to 30 minutes. In this essay, we will explore the
key factors driving the growth of quick commerce in India, the challenges
faced by companies operating in this space, and the potential future
developments that could shape the industry.

First and foremost, one of the primary drivers of quick commerce in India
is the increasing demand for convenience among consumers. With busy
lifestyles and limited time for shopping, many people are turning to quick
commerce platforms to fulfill their daily needs. Additionally, the
widespread availability of smartphones and affordable internet access has
made it easier than ever for consumers to place orders online and receive
their products quickly.

Moreover, advancements in technology have played a crucial role in enabling
the growth of quick commerce. Companies are leveraging sophisticated
algorithms and data analytics to optimize their supply chains, manage
inventory more efficiently, and predict customer demand with greater
accuracy. These technological innovations have allowed quick commerce
platforms to offer a wide range of products while maintaining fast delivery
times.

However, it is important to note that the quick commerce industry also
faces several significant challenges. One of the main concerns is the high
cost of operations, particularly when it comes to maintaining a network of
dark stores and delivery personnel. Additionally, there are concerns about
the environmental impact of rapid delivery services, as well as the working
conditions of delivery partners.

In conclusion, quick commerce represents a significant shift in the way
consumers shop for everyday products in India. While the industry has
experienced remarkable growth in recent years, it will be interesting to
see how companies navigate the challenges ahead and continue to innovate in
order to meet the evolving needs of their customers.
"""

# sub_009 (DOCX) - original, shares only the common prompt header.
HOSPITAL_ANALYSIS = CASE_PROMPT + """

Note: this candidate analysed the alternate hospital-operations brief
instead of the GreenCharge prompt; content below is original analysis.

Outpatient queues in large public hospitals are primarily a scheduling
problem, not a capacity problem. Arrival data from comparable institutions
shows sixty percent of patients arriving before 9 AM regardless of
appointment slot, which compresses waiting into the morning and leaves
afternoon capacity idle. The proposed intervention is a two-sided system:
staggered appointment slots with real-time queue visibility for patients,
and a token-release mechanism that matches doctor availability to actual
arrival patterns.

The model projects mean waiting time falling from ninety-six minutes to
thirty-eight minutes. The critical assumption is compliance: at least
seventy percent of patients must accept assigned slots. Pilots in similar
settings reached this threshold within eight weeks once SMS reminders
included a live queue estimate.

Implementation cost is modest - a scheduling service, SMS gateway, and
dashboards - while the primary risk is clinical staff resistance to
restructured session times. A phased rollout beginning with two
departments is recommended, with waiting-time p50 and p90 reviewed weekly.
"""

# sub_010 (DOCX) - original, shares only the common prompt header.
EDTECH_ANALYSIS = CASE_PROMPT + """

Note: this candidate analysed the alternate edtech brief; the analysis
below is independent original work.

Gamification in exam-preparation platforms works when rewards track
mastery rather than activity. Streaks and badges tied to daily logins
produce engagement inflation: students open the app, answer trivial
questions, and close it. The proposed design ties progression to
spaced-repetition performance, so points accrue only when a concept is
recalled correctly after increasing intervals.

The design borrows from deliberate-practice research. Each concept maps to
a skill tree node; unlocking a node requires three correct recalls spaced
over a week. Leaderboards rank classmates on retention-adjusted progress,
not raw attempts, which removes the incentive to grind easy questions.

Risks include demographic skew - competitive mechanics motivate some
students and discourage others - and reward fatigue after the first
academic term. The proposal mitigates both with cooperative team goals
that reset each month and with teacher dashboards that flag students whose
activity is high but retention is low.
"""

# Spoken text for sub_011.mp4: the "signature" paragraph of sub_001, read aloud.
MP4_CROSSMODAL_SPEECH = (
    "The core competitive advantage relies on vertical integration of the "
    "energy procurement layer. By contracting solar capacity directly and "
    "operating its own battery buffering, GreenCharge can reduce effective "
    "energy cost per kilowatt hour by roughly eighteen percent relative to "
    "grid purchasing. Operating margins expand accordingly, and the savings "
    "fund the software layer that drivers actually experience: reliable "
    "availability prediction, transparent queue times, and one tap payment. "
    "Competitors that resell grid power at retail margins cannot match this "
    "cost structure without owning generation assets."
)

# Spoken text for sub_013.mp4: original short pitch.
MP4_ORIGINAL_SPEECH = (
    "Our platform helps college fest committees sell merchandise without "
    "handling cash. Students pre-order on the app, pay through UPI, and "
    "collect from a single campus pickup point. Committees get a live "
    "dashboard of sizes and quantities, so they print exactly what is "
    "ordered and never sit on unsold stock. We take a five percent fee and "
    "handle refunds automatically. In our pilot at two campuses we "
    "processed four hundred orders in nine days with zero cash handling."
)

# sub_002 (PDF) - original GreenCharge-operations angle, shares only the prompt.
FARMFRESH_ANALYSIS = CASE_PROMPT + """

This submission examines GreenCharge's charging-station operations rather
than its market entry sequencing. The operational bottleneck is not charger
hardware but session turnover: the average fast-charging session lasts
forty-one minutes while drivers occupy the bay for fifty-seven. The
sixteen-minute gap - drivers returning late from nearby shops - is the
largest recoverable capacity source in the network.

We recommend a graduated idle-fee structure paired with accurate
charge-complete notifications. Modelling on comparable networks suggests a
fee that begins five minutes after session completion recovers roughly half
the lost throughput while keeping customer satisfaction scores within two
points of the baseline. Notifications must predict completion time within
three minutes to be trusted; this is achievable from live charging-curve
telemetry.

A second operational lever is predictive maintenance. Station downtime
concentrates in connector wear and payment-terminal faults, both of which
show telemetry signatures days before failure. A simple gradient-boosted
classifier on session logs can flag at-risk connectors with enough lead
time to batch maintenance visits, cutting truck rolls by an estimated
thirty percent.

Capital should therefore flow first to software and maintenance tooling,
with geographic expansion funded only after bay-turnover gains are
verified in two pilot stations over one quarter.
"""
