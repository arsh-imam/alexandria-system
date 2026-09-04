"""
LCATRS Archive Registry — the federated deep-archive tier.

Each entry: glob pattern (date-proof), routing description (embedded
once at load; query-to-description similarity selects specialty
archives), provenance label (shown in Sources so users see what KIND
of evidence grounded the answer). Wikipedia is always searched and
never subject to routing — current behavior is untouchable.
"""
import glob
import os
import numpy as np

ZX = "/media/pi/KINGSTON/local_ai/zim_extra/"
WP = "/media/pi/KINGSTON/local_ai/wikipedia/"

ARCHIVES = {
 "wikipedia":   {"pat": WP + "wikipedia_en*.zim", "always": True,
                 "label": "wikipedia (deep)",
                 "desc": "general encyclopedia: people, places, history, science, concepts, events, culture"},
 "medicine":    {"pat": ZX + "wikipedia_en_medicine*.zim",
                 "label": "wikipedia medicine",
                 "desc": "health, diseases, symptoms, medications, drugs, treatments, anatomy, surgery, mental health, medical conditions"},
 "wikiversity": {"pat": ZX + "wikiversity_en*.zim",
                 "label": "wikiversity",
                 "desc": "learning materials, lessons, courses, tutorials, teaching, education, exercises, study guides"},
 "skeptics":    {"pat": ZX + "skeptics.stackexchange.com*.zim",
                 "label": "skeptics SE (community Q&A)",
                 "desc": "fact-checking popular claims, myths, misconceptions, urban legends, is it true that, debunking"},
 "diy":         {"pat": ZX + "diy.stackexchange.com*.zim",
                 "label": "diy SE (community Q&A)",
                 "desc": "home improvement, repair, plumbing, electrical wiring, walls, paint, tools, renovation, fixing things at home"},
 "superuser":   {"pat": ZX + "superuser.com*.zim",
                 "label": "superuser (community Q&A)",
                 "desc": "computer problems, windows, mac, linux, software, hardware, wifi, troubleshooting, files, settings"},
 "cooking":     {"pat": ZX + "cooking.stackexchange.com*.zim",
                 "label": "cooking SE (community Q&A)",
                 "desc": "cooking, recipes, baking, ingredients, kitchen techniques, food storage, flavor, knives, ovens"},
 "travel":      {"pat": ZX + "travel.stackexchange.com*.zim",
                 "label": "travel SE (community Q&A)",
                 "desc": "travel, visas, flights, airports, luggage, hotels, border crossings, tourist destinations, itineraries"},
 "money":       {"pat": ZX + "money.stackexchange.com*.zim",
                 "label": "personal finance SE (community Q&A)",
                 "desc": "personal finance, savings, investing, taxes, credit cards, loans, budgeting, retirement, insurance"},
 "law":         {"pat": ZX + "law.stackexchange.com*.zim",
                 "label": "law SE (community Q&A)",
                 "desc": "legal questions, contracts, rights, criminal law, civil law, copyright, employment law, courts"},
 "mechanics":   {"pat": ZX + "mechanics.stackexchange.com*.zim",
                 "label": "mechanics SE (community Q&A)",
                 "desc": "cars, motor vehicles, engines, brakes, repairs, maintenance, oil, tires, batteries, transmissions"},
 "fitness":     {"pat": ZX + "fitness.stackexchange.com*.zim",
                 "label": "fitness SE (community Q&A)",
                 "desc": "exercise, workouts, strength training, running, muscles, stretching, gym, weight loss, endurance"},
}

ARCHIVES.update({
 "parenting":   {"pat": ZX + "parenting.stackexchange.com*.zim",
                 "label": "parenting SE (community Q&A)",
                 "desc": "parenting, babies, children, toddlers, teenagers, school, discipline, sleep, feeding"},
 "pets":        {"pat": ZX + "pets.stackexchange.com*.zim",
                 "label": "pets SE (community Q&A)",
                 "desc": "pets, dogs, cats, pet behavior, pet food, training animals, fish, birds, veterinary care"},
 "lifehacks":   {"pat": ZX + "lifehacks.stackexchange.com*.zim",
                 "label": "lifehacks SE (community Q&A)",
                 "desc": "everyday tricks, cleaning, organizing, household shortcuts, stains, storage, quick fixes"},
 "interpersonal": {"pat": ZX + "interpersonal.stackexchange.com*.zim",
                 "label": "interpersonal SE (community Q&A)",
                 "desc": "social skills, communication, conflict, friendships, colleagues, awkward situations, etiquette"},
 "psychology":  {"pat": ZX + "psychology.stackexchange.com*.zim",
                 "label": "psychology SE (community Q&A)",
                 "desc": "psychology, neuroscience, cognition, behavior, memory, emotions, mental processes, brain"},
 "philosophy":  {"pat": ZX + "philosophy.stackexchange.com*.zim",
                 "label": "philosophy SE (community Q&A)",
                 "desc": "philosophy, ethics, logic, metaphysics, epistemology, philosophers, meaning, morality, free will"},
 "movies":      {"pat": ZX + "movies.stackexchange.com*.zim",
                 "label": "movies SE (community Q&A)",
                 "desc": "movies, films, TV shows, plots, endings, directors, scenes, characters, film techniques"},
 "anime":       {"pat": ZX + "anime.stackexchange.com*.zim",
                 "label": "anime SE (community Q&A)",
                 "desc": "anime, manga, episodes, characters, Japanese animation, series, plot questions"},
 "literature":  {"pat": ZX + "literature.stackexchange.com*.zim",
                 "label": "literature SE (community Q&A)",
                 "desc": "books, novels, poetry, authors, literary analysis, characters, themes, classic literature"},
 "musicfans":   {"pat": ZX + "musicfans.stackexchange.com*.zim",
                 "label": "music fans SE (community Q&A)",
                 "desc": "music, bands, songs, albums, artists, lyrics meaning, genres, concerts"},
 "sports":      {"pat": ZX + "sports.stackexchange.com*.zim",
                 "label": "sports SE (community Q&A)",
                 "desc": "sports rules, football, cricket, tennis, basketball, athletes, scoring, regulations, records"},
 "boardgames":  {"pat": ZX + "boardgames.stackexchange.com*.zim",
                 "label": "board games SE (community Q&A)",
                 "desc": "board games, card games, chess, rules, strategy, tabletop games"},
 "writing":     {"pat": ZX + "writing.stackexchange.com*.zim",
                 "label": "writing SE (community Q&A)",
                 "desc": "writing craft, fiction, essays, style, publishing, grammar in writing, storytelling, editing"},
 "academia":    {"pat": ZX + "academia.stackexchange.com*.zim",
                 "label": "academia SE (community Q&A)",
                 "desc": "universities, research, PhD, professors, papers, citations, peer review, academic careers, exams"},
})

ARCHIVES.update({
 "economics":   {"pat": ZX + "economics.stackexchange.com*.zim",
                 "label": "economics SE (community Q&A)",
                 "desc": "economics, markets, inflation, GDP, monetary policy, trade, supply and demand, macroeconomics"},
 "quant":       {"pat": ZX + "quant.stackexchange.com*.zim",
                 "label": "quantitative finance SE (community Q&A)",
                 "desc": "quantitative finance, options, derivatives, pricing models, portfolios, trading, risk"},
 "engineering": {"pat": ZX + "engineering.stackexchange.com*.zim",
                 "label": "engineering SE (community Q&A)",
                 "desc": "engineering, mechanical, civil, structures, materials, electrical engineering, design, machines"},
 "earthscience": {"pat": ZX + "earthscience.stackexchange.com*.zim",
                 "label": "earth science SE (community Q&A)",
                 "desc": "geology, weather, climate, earthquakes, volcanoes, oceans, atmosphere, minerals, natural phenomena"},
 "politics":    {"pat": ZX + "politics.stackexchange.com*.zim",
                 "label": "politics SE (community Q&A)",
                 "desc": "politics, governments, elections, policies, international relations, political systems, laws"},
 "christianity": {"pat": ZX + "christianity.stackexchange.com*.zim",
                 "label": "christianity SE (community Q&A)",
                 "desc": "Christianity, Bible, theology, churches, Christian beliefs, denominations, scripture"},
 "islam":       {"pat": ZX + "islam.stackexchange.com*.zim",
                 "label": "islam SE (community Q&A)",
                 "desc": "Islam, Quran, hadith, Islamic practice, prayer, fasting, Islamic beliefs and law"},
 "hinduism":    {"pat": ZX + "hinduism.stackexchange.com*.zim",
                 "label": "hinduism SE (community Q&A)",
                 "desc": "Hinduism, Vedas, deities, Hindu philosophy, rituals, scriptures, dharma, traditions"},
 "crafts":      {"pat": ZX + "crafts.stackexchange.com*.zim",
                 "label": "arts and crafts SE (community Q&A)",
                 "desc": "crafts, knitting, sewing, woodworking, painting techniques, glue, handmade projects"},
})




# Epistemic type per archive (additive metadata; does not affect
# routing or retrieval — consumed only by optional epistemic fusion).
_ETYPES = {'wikipedia': 'encyclopedic', 'medicine': 'encyclopedic', 'wikiversity': 'instructional', 'skeptics': 'debunking', 'diy': 'instructional', 'superuser': 'instructional', 'cooking': 'instructional', 'travel': 'community', 'money': 'community', 'law': 'community', 'mechanics': 'instructional', 'fitness': 'instructional', 'parenting': 'community', 'pets': 'community', 'lifehacks': 'instructional', 'interpersonal': 'community', 'psychology': 'community', 'philosophy': 'community', 'movies': 'community', 'anime': 'community', 'literature': 'community', 'musicfans': 'community', 'sports': 'community', 'boardgames': 'community', 'writing': 'instructional', 'academia': 'community', 'economics': 'community', 'quant': 'community', 'engineering': 'community', 'earthscience': 'community', 'politics': 'community', 'christianity': 'community', 'islam': 'community', 'hinduism': 'community', 'crafts': 'instructional'}
for _n, _cfg in ARCHIVES.items():
    _cfg["etype"] = _ETYPES.get(_n, "encyclopedic")


class ArchiveRegistry:

    def __init__(self):
        self.entries = {}      # name -> {archive, label, desc, always}
        self.route_names = []  # routable (non-always) archive names
        self.route_vecs = None

    def load(self, embedder):
        from libzim.reader import Archive
        missing = []
        for name, cfg in ARCHIVES.items():
            hits = sorted(glob.glob(cfg["pat"]))
            if not hits:
                missing.append(name)
                continue
            try:
                arch = Archive(hits[-1])
            except Exception as e:
                print(f"  registry: {name} failed to open ({e})")
                continue
            self.entries[name] = {
                "archive": arch,
                "label": cfg["label"],
                "desc": cfg["desc"],
                "always": cfg.get("always", False),
                "etype": cfg.get("etype", "encyclopedic"),
                "file": os.path.basename(hits[-1]),
            }
        routable = [n for n, e in self.entries.items() if not e["always"]]
        if routable:
            vecs = embedder.embed([self.entries[n]["desc"]
                                   for n in routable])
            self.route_names = routable
            self.route_vecs = np.asarray(vecs, dtype=np.float32)
        print(f"  registry: {len(self.entries)} archives open "
              f"({len(routable)} routable)"
              + (f", missing: {missing}" if missing else ""))
        return self

    def route(self, query_vec, k=3):
        """Top-k specialty archives by query-description similarity."""
        if self.route_vecs is None:
            return []
        sims = self.route_vecs @ np.asarray(query_vec, dtype=np.float32)
        order = np.argsort(-sims)[:k]
        return [(self.route_names[i], float(sims[i])) for i in order]

    def always_archives(self):
        return [n for n, e in self.entries.items() if e["always"]]

    def get(self, name):
        return self.entries.get(name)
