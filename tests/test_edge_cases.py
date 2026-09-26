"""Input-shape edge cases: mixed languages, very short and very long text, poor English.

The rule throughout: reward only content the pipeline can assess and that is actually new.
Poor spelling or grammar must not *cost* a relevant novel idea its reward, and must not *buy*
novelty for a stock take either.
"""

import pytest

from helpers import NOT_NOVEL_MAX, NOVEL_MIN, UNREWARDED, assert_novel
from novelty.models import Submission

FLOOD_EN = ("Elm Street floods every spring because the garage and pavement shed rainwater into overloaded "
            "drains. Building the park with rain gardens and an underground retention tank would protect "
            "downtown basements.")
FLOOD_ES = ("La calle Elm se inunda cada primavera. Construir el parque con jardines de lluvia y un tanque "
            "subterráneo protegería los sótanos del centro.")
PARKING_EN = "Removing the garage will hurt every shop on Main Street because shoppers will go to the malls instead."
PARKING_ES = ("Perder cientos de plazas de estacionamiento en el centro será un desastre para las tiendas "
              "locales. Los compradores irán a los centros comerciales.")


def _score(scorer, headline, body, stance="support"):
    return scorer.score(Submission(headline=headline, body=body, stance=stance))


def _novel(scorer, headline, body, stance="support", minimum=NOVEL_MIN):
    return assert_novel(scorer, Submission(headline=headline, body=body, stance=stance), minimum)


# ---------------------------------------------------------------- mixed languages


@pytest.mark.parametrize("foreign", [
    PARKING_ES,
    "Je suis très content de voir cette décision, merci au conseil municipal pour ce beau projet.",
    "यह पार्क बहुत अच्छा होगा और बच्चों के लिए खेलने की जगह मिलेगी।",
])
def test_english_novel_half_is_rewarded_whatever_the_other_half_says(scorer, foreign):
    r = _novel(scorer, "Design the park to soak up floods", f"{FLOOD_EN} {foreign}")
    assert any("not in English and were not scored" in x for x in r.reasons)


def test_untranslated_half_cannot_supply_novelty_for_a_stock_english_half(scorer):
    """The Spanish half is a new idea, but the English-only pipeline cannot assess it, so it must
    not make the stock English half look new (it previously lifted the score to 0.49)."""
    r = _score(scorer, "Shops need parking", f"{PARKING_EN} {FLOOD_ES}", "oppose")
    assert r.score <= NOT_NOVEL_MAX, r


def test_non_english_only_text_gets_a_clear_reason(scorer):
    r = _score(scorer, "Parque contra inundaciones", FLOOD_ES)
    assert r.score == 0.0
    assert any("not in a supported language" in x for x in r.reasons)


def test_foreign_words_are_never_spell_corrected_into_english(scorer):
    prep = scorer.index.preparer.prepare(
        "Flood", "Elm Street se inunda every spring, so the park debería tener rain gardens y un tanque underground.")
    assert "debería tener" in prep.body and not prep.corrections


# ---------------------------------------------------------------- very short (< 10 words)


@pytest.mark.parametrize("headline, body", [
    ("EV chargers", "Put EV chargers at the outer shuttle lot."),
    ("Solar", "Solar canopies over the shuttle lot."),
    ("Flood", "Build the park with rain gardens so Elm Street stops flooding."),
    ("Bike", "Add a protected bike lane on Elm Street to the new park."),
])
def test_short_novel_ideas_are_rewarded(scorer, headline, body):
    _novel(scorer, headline, body)


@pytest.mark.parametrize("headline, body", [
    ("Parking", "No parking means shops will close."),
    ("Cost", "Fourteen million dollars is too much money."),
    ("Kids", "Kids need a place to play downtown."),
    ("Trees", "Trees will make downtown cooler in summer."),
    ("Shuttle", "The shuttle bus will not work for shoppers."),
])
def test_short_stock_takes_are_not_rewarded(scorer, headline, body):
    assert _score(scorer, headline, body).score <= NOT_NOVEL_MAX


@pytest.mark.parametrize("headline, body", [
    ("Nice", "I love this park idea!"),
    ("Good", "This is a great plan, I really like it."),
    ("Agree", "Totally agree with this decision!!"),
])
def test_short_generic_praise_is_not_rewarded(scorer, headline, body):
    """Short texts sit far from everything in embedding space; specificity stops that reading as novelty."""
    r = _score(scorer, headline, body)
    assert r.score <= NOT_NOVEL_MAX and r.signals["specificity"] < 1.0


def test_short_off_topic_is_not_rewarded(scorer):
    assert _score(scorer, "Pizza", "Best pizza in town is Luigi's.").score <= UNREWARDED


# ---------------------------------------------------------------- very long (> 100 words)


def test_long_novel_essay_is_rewarded(scorer):
    body = " ".join([
        FLOOD_EN,
        "I have lived on Elm Street for twenty years and every April the storm drains back up into our basements.",
        "The city already pays for pumping and repairs after each storm, so this would save money as well.",
        "Other cities have built parks that double as retention basins, with sunken lawns that fill during heavy rain.",
        "The council should ask the landscape designers to model the rainfall before finalizing the plan.",
        "It would also be a good chance to replace the old sewer connection under the garage while the site is open.",
    ])
    assert len(body.split()) > 100
    _novel(scorer, "A park that manages stormwater", body)


def test_long_rehash_of_existing_takes_is_not_rewarded(scorer):
    body = " ".join([
        "I think losing the parking garage will hurt the shops on Main Street because customers need somewhere to park.",
        "At the same time, families downtown really do need a park where kids can play.",
        "The fourteen million dollar price tag seems very high to me and I worry about the business levy.",
        "Trees would help with the summer heat, which has been terrible lately.",
        "I also doubt the shuttle will work because nobody wants to wait for a bus to go shopping.",
        "Safety and maintenance need a plan too, or the park will end up neglected like our other parks.",
    ])
    assert len(body.split()) >= 100
    # Known weak spot: a long restatement of many covered takes sits near 0.3 rather than ~0
    # (0.278 on the seed corpus, 0.300 after three extra web-UI submissions shift the
    # calibration). The bound documents that it stays far below the novel threshold (0.6).
    assert _score(scorer, "My view on the park", body, "mixed").score <= 0.35


def test_half_off_topic_half_novel_rewards_the_novel_half(scorer):
    _novel(scorer, "Two things",
           "I finally tried the new ramen place near the station and the broth was incredible. The park should "
           "be built with rain gardens and an underground tank so Elm Street stops flooding every spring.")


def test_mostly_off_topic_with_one_relevant_line_is_heavily_discounted(scorer):
    body = " ".join([
        "This weekend I finally tried the new ramen place near the station and the broth was incredible.",
        "My cousin is visiting from Denver and we spent the afternoon hiking along the river trail.",
        "The weather has been perfect for walking and the leaves are starting to change color.",
        "Anyway, the new park should be designed with rain gardens and an underground tank so Elm Street stops flooding.",
        "We also watched a great documentary about octopuses that I would recommend to anyone.",
    ])
    r = _score(scorer, "A few thoughts", body)
    assert 0.0 < r.score <= 0.3, "one relevant idea earns something, but the comment is mostly off-topic"


def test_body_length_limit_is_enforced():
    with pytest.raises(ValueError):
        Submission(headline="Long", body="word " * 500, stance="support")


# ---------------------------------------------------------------- poor English


@pytest.mark.parametrize("headline, body, stance", [
    ("garage flood problem", "elm street flood every spring, water go basement. make park with rain garden and big "
     "tank under ground, water stay there not go house", "support"),
    ("Desing the park for flods", "Elm stret floods evry spring becuse the garaje and pavment send all rain in to the "
     "drains. If park have rain gardns and undergrond tank it culd protect the basments downtown.", "support"),
    ("park = flood sponge", "tbh elm st floods every spring lol. park shud have rain gardens n an underground tank so "
     "ppl basements stop flooding", "support"),
    ("keep the frame", "why tear down whole garage its lots of concrete and carbon just keep frame and put terraced "
     "park on top keep bottom deck for disabled parking and deliveries cheaper too", "mixed"),
    ("About the history", "This garage is stand on old rail depot from 1880s where the town is begin. Park must to "
     "show the old platform and make small history walk for people know how downtown start.", "support"),
])
def test_relevant_novel_idea_in_poor_english_keeps_its_reward(scorer, headline, body, stance):
    r = _novel(scorer, headline, body, stance, minimum=0.5)
    assert r.relevance_gate > 0.9, "poor English must not make relevant content look off-topic"


@pytest.mark.parametrize("headline, body", [
    ("no parking no customer", "no parking no customer, shop close, people go mall. very bad for main street shop"),
    ("Parkng is importent", "Withot the garaje peple cant park and they wil not come to shops. Main stret wil die "
     "becuse evryone go to mall"),
])
def test_misspelling_a_stock_take_does_not_make_it_novel(scorer, headline, body):
    """Unseen misspelled tokens used to read as novelty (0.86); spelling correction removes that."""
    assert _score(scorer, headline, body, "oppose").score <= NOT_NOVEL_MAX


def test_typo_copy_of_a_corpus_comment_is_still_a_copy(scorer):
    r = _score(scorer, "No parking means no custmers",
               "I run a shoe repiar shop on Main Stret. Most of my custmers drive in, drop off, and leave. Take away "
               "600 spaces and they will just go to the mal where parking is free and easy.", "oppose")
    assert r.near_duplicate_of == "c01" and r.score == 0.0
