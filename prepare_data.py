#!/usr/bin/env python3
"""Data curation pipeline: collect, format, blend (40/40/20), and split.

Produces train.jsonl and val.jsonl ready for Unsloth fine-tuning.
"""

import json
import random
import argparse
from pathlib import Path

SEED = 42

# ---------------------------------------------------------------------------
# Raw data sources — expand these lists or load from external files.
# ---------------------------------------------------------------------------

DICTIONARY_ENTRIES = [
    {"word": "abide", "definition": "To accept or act in accordance with a rule or decision.", "example": "We must abide by the rules of the competition."},
    {"word": "azure", "definition": "A bright blue color resembling a cloudless sky.", "example": "The azure waters of the Mediterranean sparkled in the sunlight."},
    {"word": "breeze", "definition": "A gentle wind.", "example": "A cool breeze swept through the open window."},
    {"word": "brook", "definition": "A small stream.", "example": "The children played beside the babbling brook all afternoon."},
    {"word": "candle", "definition": "A cylinder of wax with a central wick that is lit to produce light.", "example": "She lit a candle and placed it on the windowsill."},
    {"word": "dawn", "definition": "The first appearance of light in the sky before sunrise.", "example": "They set out at dawn to reach the summit by noon."},
    {"word": "ember", "definition": "A small live piece of coal or wood in a dying fire.", "example": "The last embers glowed softly in the fireplace."},
    {"word": "fathom", "definition": "To understand after much thought; a unit of length equal to six feet.", "example": "I cannot fathom how she solved the puzzle so quickly."},
    {"word": "glimmer", "definition": "A faint or wavering light; a faint sign of a quality.", "example": "A glimmer of hope appeared on the horizon."},
    {"word": "hush", "definition": "To make or become quiet.", "example": "A hush fell over the audience as the conductor raised the baton."},
    {"word": "ivy", "definition": "A climbing or trailing evergreen plant.", "example": "Thick ivy covered the walls of the old university building."},
    {"word": "jubilant", "definition": "Feeling or expressing great happiness and triumph.", "example": "The jubilant crowd celebrated the team's victory."},
    {"word": "kindle", "definition": "To light or set on fire; to arouse or inspire.", "example": "Her speech kindled a desire for change in the community."},
    {"word": "lantern", "definition": "A lamp with a transparent case protecting the flame or bulb.", "example": "He carried a lantern through the dark forest path."},
    {"word": "meadow", "definition": "A piece of grassland, especially one used for hay.", "example": "Wildflowers dotted the meadow with splashes of color."},
    {"word": "nocturnal", "definition": "Active during the night.", "example": "Owls are nocturnal creatures that hunt after dark."},
    {"word": "oak", "definition": "A large tree that produces acorns and has lobed leaves.", "example": "The ancient oak stood at the center of the village green."},
    {"word": "petal", "definition": "Each of the segments of the corolla of a flower.", "example": "Rose petals scattered across the garden path."},
    {"word": "quill", "definition": "A large feather used as a pen; a spine of a porcupine.", "example": "The poet dipped the quill in ink and began to write."},
    {"word": "river", "definition": "A large natural stream of water flowing to the sea or a lake.", "example": "The river wound through the valley like a silver ribbon."},
    {"word": "solace", "definition": "Comfort or consolation in a time of distress.", "example": "She found solace in music during the long winter evenings."},
    {"word": "thistle", "definition": "A prickly plant with purple flower heads.", "example": "The thistle is the national emblem of Scotland."},
    {"word": "umbra", "definition": "The fully shaded inner region of a shadow.", "example": "During the eclipse, the umbra swept across the continent."},
    {"word": "velvet", "definition": "A closely woven fabric with a thick short pile on one side.", "example": "She wore a velvet gown the color of midnight."},
    {"word": "willow", "definition": "A tree with narrow leaves and slender drooping branches.", "example": "The willow dipped its branches into the still pond."},
    {"word": "zephyr", "definition": "A soft gentle breeze.", "example": "A warm zephyr carried the scent of blossoms through the garden."},
    {"word": "ardent", "definition": "Very enthusiastic or passionate.", "example": "He was an ardent supporter of environmental conservation."},
    {"word": "blithe", "definition": "Showing a casual and cheerful indifference.", "example": "With blithe disregard for the rules, she danced across the fountain."},
    {"word": "cascade", "definition": "A small waterfall, especially one in a series.", "example": "Water cascaded down the rocky cliff into the pool below."},
    {"word": "dusk", "definition": "The darker stage of twilight, especially in the evening.", "example": "Fireflies began to glow at dusk across the open field."},
    {"word": "ephemeral", "definition": "Lasting for a very short time.", "example": "The beauty of cherry blossoms is ephemeral, lasting only days."},
    {"word": "forge", "definition": "To create something strong or successful; a blacksmith's workshop.", "example": "They forged a lasting alliance through years of cooperation."},
    {"word": "gossamer", "definition": "A fine, filmy substance of cobwebs; something very light and insubstantial.", "example": "Gossamer threads of spider silk caught the morning dew."},
    {"word": "harbinger", "definition": "A person or thing that announces the approach of another.", "example": "The robin is often seen as a harbinger of spring."},
    {"word": "incandescent", "definition": "Emitting light as a result of being heated; passionate.", "example": "The incandescent bulb cast a warm glow across the room."},
    {"word": "labyrinth", "definition": "A complicated irregular network of passages or paths.", "example": "The old castle contained a labyrinth of underground tunnels."},
    {"word": "murmur", "definition": "A soft, indistinct sound made by a person or group.", "example": "A murmur of agreement rippled through the crowd."},
    {"word": "nimble", "definition": "Quick and light in movement or action.", "example": "The nimble cat leapt from branch to branch without hesitation."},
    {"word": "obsidian", "definition": "A hard, dark volcanic glass formed by rapid cooling of lava.", "example": "The arrowhead was carved from a piece of obsidian."},
    {"word": "pristine", "definition": "In its original condition; unspoiled.", "example": "The pristine lake reflected the snow-capped mountains perfectly."},
    {"word": "quench", "definition": "To satisfy thirst; to extinguish a fire.", "example": "Nothing could quench her thirst for knowledge."},
    {"word": "reverie", "definition": "A state of being pleasantly lost in one's thoughts.", "example": "She sat in quiet reverie, gazing at the sunset."},
    {"word": "serene", "definition": "Calm, peaceful, and untroubled.", "example": "The serene landscape offered a welcome escape from city life."},
    {"word": "talisman", "definition": "An object thought to have magic powers and to bring good luck.", "example": "He kept the old coin as a talisman against misfortune."},
    {"word": "verdant", "definition": "Green with grass or other rich vegetation.", "example": "The verdant hills rolled on endlessly under the summer sky."},
    {"word": "wistful", "definition": "Having or showing a feeling of vague or regretful longing.", "example": "She cast a wistful glance at the old photograph."},
]

POETRY_DATA = [
    # Completion prompts
    {"prompt": "Continue the poem:\nAt dawn the willow keeps the river's secret,\nA lantern hums where meadow grasses sleep.", "completion": "Night folds velvet skies above the field,\nAnd petal-light the embers start to weep."},
    {"prompt": "Continue the poem:\nThe brook runs silver under morning light,", "completion": "While ivy climbs the walls of ancient stone.\nA quill records what candles cannot write,\nAnd solace blooms in gardens overgrown."},
    {"prompt": "Continue the poem:\nBeneath the oak a thousand acorns dream,", "completion": "Of forests stretching past the farthest hill.\nThe zephyr hums a barely-whispered theme,\nAnd time itself grows gentle, calm, and still."},
    {"prompt": "Write a haiku about autumn:", "completion": "Crimson leaves descend,\nThe oak stands bare against sky—\nWind carries their song."},
    {"prompt": "Write a haiku about the sea:", "completion": "Azure waves retreat,\nLeaving shells on quiet sand—\nThe tide remembers."},
    {"prompt": "Write a couplet about dawn:", "completion": "When dawn ignites the eastern ridge with gold,\nThe meadow stirs with stories yet untold."},
    {"prompt": "Write a couplet about night:", "completion": "The night descends in velvet, dark and deep,\nAnd lantern-lit, the village falls asleep."},
    {"prompt": "Complete the stanza:\nA breeze through thistle, soft and low,\nThe ember fades, the candle's glow", "completion": "Recedes like rivers, sure but slow,\nWhile willow branches dip below."},
    {"prompt": "Write a quatrain about solitude:", "completion": "In hush of evening, far from town,\nThe poet lays her quill-pen down.\nThe brook provides the only sound—\nA murmur, constant and profound."},
    {"prompt": "Write a poem about memory:", "completion": "I found an ember in the ash of years,\nA glimmer wrapped in gossamer and dust.\nIt kindled warmth that burned away my fears\nAnd forged from rust a bright, incandescent trust."},
    {"prompt": "Write a short verse about a forest:", "completion": "Beneath the canopy of oak and pine,\nWhere ivy weaves its labyrinthine thread,\nThe nimble fox treads paths of old design,\nAnd ferns unfurl where ancient feet have led."},
    {"prompt": "Continue the poem:\nThe garden sleeps beneath a quilt of snow,", "completion": "Yet underground the verdant promise stays.\nThe bulbs remember what the blossoms know—\nThat spring will come in warm and golden days."},
    {"prompt": "Write a poem about the wind:", "completion": "The zephyr speaks in tongues of leaf and dust,\nA harbinger of storms or gentle rain.\nIn it the trees place all their quiet trust,\nAnd bend, and sway, and straighten up again."},
    {"prompt": "Write a sonnet's opening about twilight:", "completion": "When dusk descends and paints the sky in rose,\nThe world grows soft and shadows start to blend.\nThe murmur of the day draws to a close,\nAnd starlight marks the evening's gentle end."},
    {"prompt": "Complete this poem:\nA wistful glance across the autumn field,", "completion": "Where once we walked through meadows, hand in hand.\nThe harvest done, the final crop now yields\nTo frost that claims the tired and grateful land."},
    {"prompt": "Write a verse about a river journey:", "completion": "The river carries secrets to the sea,\nPast willow banks and bridges built of stone.\nIt asks for nothing, flowing wild and free,\nYet gives to all the lands through which it's flown."},
    {"prompt": "Write two lines about candles:", "completion": "A candle is a patient, gentle thing—\nIt gives its life to light, yet does not cling."},
    {"prompt": "Write a verse about stars:", "completion": "Each star a lantern hung in endless night,\nA glimmer in the obsidian expanse.\nThey do not speak, yet fill the world with light,\nAnd guide the lost through fortune and through chance."},
    {"prompt": "Write a tanka about rain:", "completion": "Soft rain on the lake,\nEach drop a tiny cascade—\nRipples touch the shore,\nThen the hush of aftermath,\nAnd the pristine sky returns."},
    {"prompt": "Continue the poem:\nWhere embers cool and morning light arrives,", "completion": "The forge grows silent, tools upon the wall.\nYet in the steel the maker's art survives,\nA talisman against the coming fall."},
]

LITTLE_CORPUS = [
    {"text": "The apprentice writes a short letter with a careful hand. Every word is chosen to carry weight without waste. Brevity in writing is a form of respect for the reader's time."},
    {"text": "A small language model can still learn rhythm and pattern from tiny books. The key is clean data and consistent formatting. Quality matters more than quantity at small scales."},
    {"text": "The village keeps a dictionary beside the old poems. Children learn new words each morning and recite verses each evening. Language grows best when vocabulary and expression develop together."},
    {"text": "Simple systems improve when examples are clean and focused. Noise in training data creates noise in output. Careful curation is the most underrated step in machine learning."},
    {"text": "Iterative editing can polish rough text into clear phrases. Each pass removes unnecessary words and sharpens meaning. The best writing is rewriting."},
    {"text": "Fine-tuning a language model is like teaching a student who already knows grammar. You don't start from scratch—you build on existing knowledge with targeted examples."},
    {"text": "LoRA adapters add a small number of trainable parameters on top of a frozen base model. This makes fine-tuning fast and memory-efficient, even on consumer hardware."},
    {"text": "A dictionary entry pairs a word with its meaning and usage. This structured format teaches a model the relationship between form and function in language."},
    {"text": "Poetry teaches rhythm, metaphor, and compression. A model trained on verse learns that language can be dense with meaning and still flow naturally."},
    {"text": "The blended dataset approach mixes structured data like dictionary entries with creative text like poetry. This teaches both precision and fluency."},
    {"text": "Evaluation of language models should combine automatic metrics like perplexity with human judgment of generation quality. Numbers alone cannot capture whether text feels right."},
    {"text": "Small models excel when given a narrow domain. Rather than trying to know everything, a focused model can achieve surprising quality on its specific task."},
    {"text": "The learning rate controls how fast a model updates its weights. Too high and it overshoots; too low and training stalls. Finding the right balance requires experimentation."},
    {"text": "Gradient accumulation allows training with larger effective batch sizes on limited hardware. It sums gradients over multiple forward passes before updating weights."},
    {"text": "Warmup steps gradually increase the learning rate at the start of training. This prevents the model from making wild updates before it has seen enough data to learn stable patterns."},
    {"text": "A held-out validation set is essential for detecting overfitting. If training loss drops but validation loss rises, the model is memorizing rather than learning."},
    {"text": "Tokenization breaks text into subword units that the model processes. Different tokenizers produce different splits, which affects how the model represents and generates language."},
    {"text": "The attention mechanism lets a model focus on relevant parts of the input when generating each output token. It is the core innovation behind transformer architectures."},
    {"text": "Quantization reduces model precision from 32-bit to 4-bit floats, dramatically cutting memory usage. With techniques like QLoRA, the quality loss is minimal."},
    {"text": "A well-crafted prompt template ensures consistent formatting during both training and inference. Mismatched formats between training and generation degrade output quality."},
    {"text": "Transfer learning works because language has universal patterns. A model trained on general text already understands syntax, semantics, and common sense before fine-tuning begins."},
    {"text": "The diffusion approach to language generation treats text as a signal to be iteratively refined. Starting from noise, each step brings the output closer to coherent language."},
    {"text": "Perplexity measures how surprised a model is by the test data. Lower perplexity means better prediction. It is the standard automatic metric for language model quality."},
]


def format_dictionary_entry(entry: dict) -> str:
    """Format a dictionary entry into training text."""
    return f"Word: {entry['word']}\nDefinition: {entry['definition']}\nExample: {entry['example']}"


def format_poetry(entry: dict) -> str:
    """Format a poetry prompt-completion pair into training text."""
    return f"{entry['prompt']}\n{entry['completion']}"


def format_corpus(entry: dict) -> str:
    """Format a corpus paragraph into training text."""
    return entry["text"]


def build_dataset(
    dict_entries: list[dict],
    poetry_entries: list[dict],
    corpus_entries: list[dict],
    val_ratio: float = 0.1,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """Blend data 40/40/20 and split into train/val."""
    rng = random.Random(seed)

    # Format all entries
    dict_samples = [{"text": format_dictionary_entry(e), "source": "dictionary"} for e in dict_entries]
    poetry_samples = [{"text": format_poetry(e), "source": "poetry"} for e in poetry_entries]
    corpus_samples = [{"text": format_corpus(e), "source": "corpus"} for e in corpus_entries]

    # To achieve ~40/40/20 blend, we oversample smaller categories
    # Target: equal effective weight per category proportional to 40/40/20
    n_dict = len(dict_samples)
    n_poetry = len(poetry_samples)
    n_corpus = len(corpus_samples)

    # Use the raw counts — the natural ratio here is roughly 46/20/23 = ~50/22/26
    # We'll adjust by repeating smaller sets to approximate 40/40/20
    # Target ratio: dict:poetry:corpus = 2:2:1
    # Scale poetry up to match dict count, corpus to half of dict
    target_poetry = n_dict  # match dictionary count for 40/40
    target_corpus = n_dict // 2  # half for 20%

    poetry_repeated = (poetry_samples * ((target_poetry // n_poetry) + 1))[:target_poetry]
    corpus_repeated = (corpus_samples * ((target_corpus // n_corpus) + 1))[:target_corpus]

    all_samples = dict_samples + poetry_repeated + corpus_repeated
    rng.shuffle(all_samples)

    # Split
    val_count = max(1, int(len(all_samples) * val_ratio))
    val_set = all_samples[:val_count]
    train_set = all_samples[val_count:]

    return train_set, val_set


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for small LLM fine-tuning")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for train/val JSONL files")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_dataset(
        DICTIONARY_ENTRIES, POETRY_DATA, LITTLE_CORPUS,
        val_ratio=args.val_ratio, seed=args.seed,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    save_jsonl(train_set, train_path)
    save_jsonl(val_set, val_path)

    # Print statistics
    sources_train = {}
    for item in train_set:
        sources_train[item["source"]] = sources_train.get(item["source"], 0) + 1
    sources_val = {}
    for item in val_set:
        sources_val[item["source"]] = sources_val.get(item["source"], 0) + 1

    total_train = len(train_set)
    print(f"=== Data Preparation Complete ===")
    print(f"Train samples: {total_train}")
    for src, count in sorted(sources_train.items()):
        print(f"  {src}: {count} ({100*count/total_train:.1f}%)")
    print(f"Val samples:   {len(val_set)}")
    for src, count in sorted(sources_val.items()):
        print(f"  {src}: {count}")
    print(f"\nFiles written:")
    print(f"  {train_path}")
    print(f"  {val_path}")


if __name__ == "__main__":
    main()
