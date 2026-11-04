"""Text lists for the Neve wake-word spike: positives and hard negatives."""

import itertools
import random

# Positives: Italian spelling (Italian voices) and raw IPA (other voices, closed and open e).
POS_IT_TEXT = [
    "Neve.",
    "Neve!",
    "Neve?",
    "Ehi Neve!",
    "Ehi, Neve.",
    "Ciao Neve!",
    "Ciao, Neve.",
    "Ehi Neve?",
]
POS_IPA_IT = ["nˈeve", "nˈɛve", "ˈei nˈeve", "ˈei nˈɛve", "tʃˈao nˈeve", "tʃˈao nˈɛve", "ˈɛi nˈeve"]
POS_EN_TEXT = ["Nevay.", "Hey Nevay!", "Hi Nevay.", "Chow Nevay!", "Ciao Nevay."]
POS_IPA_EN = ["nˈɛveɪ", "nˈeɪveɪ", "hˈeɪ nˈɛveɪ", "hˈeɪ nˈɛve", "tʃˈaʊ nˈɛve", "tʃˈaʊ nˈɛveɪ"]

# Hard negatives, Italian: "neve" inside a sentence, its family, and near words.
NEVE_SENT = [
    "C'è la neve.",
    "Guarda, c'è la neve!",
    "Domani arriva la neve.",
    "La neve è bianca e fredda.",
    "Ieri ho visto la neve.",
    "Sulle montagne c'è ancora neve.",
    "Abbiamo fatto un pupazzo di neve.",
    "La neve copre tutto il giardino.",
    "Quanta neve è caduta stanotte?",
    "Mi piace camminare sulla neve.",
    "Le strade sono piene di neve.",
    "Niente neve quest'anno, purtroppo.",
    "Andiamo a sciare sulla neve fresca.",
    "Bianco come la neve.",
    "La neve si scioglie al sole.",
    "Hanno chiuso la scuola per la neve.",
    "Il gatto gioca con la neve.",
    "Una palla di neve mi ha colpito.",
    "Neve e ghiaccio sull'autostrada.",
    "Previsioni: neve in collina.",
    "Ho sognato la neve a Natale.",
    "Le catene da neve sono in macchina.",
    "Fiocchi di neve dappertutto.",
    "Che bella la neve!",
    "La neve non è ancora arrivata.",
    "In città la neve dura poco.",
    "Biancaneve e i sette nani.",
    "Abbiamo spalato la neve dal vialetto.",
    "Il rumore della neve sotto le scarpe.",
    "Senza neve le piste sono chiuse.",
    "Nevica da stamattina.",
    "Domani nevica in montagna.",
    "Che nevicata ieri sera!",
    "Un inverno nevoso.",
    "Stanotte ha nevicato molto.",
    "Le cime innevate sono bellissime.",
    "Se nevica restiamo a casa.",
    "Il cielo è grigio, sembra che nevichi.",
    "Il nevaio sotto la vetta.",
    "Una nevicata improvvisa.",
    "Neve, neve, ancora neve.",
    "Parliamo della neve di ieri.",
    "La neve rende tutto silenzioso.",
    "Ho comprato gli stivali da neve.",
    "Il bambino disegna la neve.",
    "Non vedo l'ora che arrivi la neve.",
    "Era tutto coperto di neve.",
    "Ci vuole la neve per sciare.",
    "La neve cade lenta.",
    "Odio guidare con la neve.",
]
NEAR_SENT = [
    "Devi andare adesso.",
    "Lui deve partire domani.",
    "Beve un bicchiere d'acqua.",
    "Una pausa breve.",
    "La nave è in porto.",
    "Ciao Eva, come stai?",
    "Ehi Eva, vieni qui!",
    "Un vento lieve.",
    "Che caso grave.",
    "Il neo sulla guancia.",
    "Ciao Nello!",
    "Ehi Nevio, ci sei?",
    "Ho i nervi a pezzi.",
    "Ciao Nadia.",
    "Ciao Neri, a domani.",
    "Neanche per sogno.",
    "Nero come la notte.",
    "Ne ho visti tre.",
    "Ehi, vieni a vedere.",
    "Ciao a tutti!",
    "Ciao, ci vediamo dopo.",
    "Ehi tu, aspetta!",
    "Ciao bella!",
    "Ehi, che succede?",
    "Nevrotico, direi.",
    "Vede tutto lui.",
    "Ce n'è ancora.",
    "La neve? No, la nave.",
    "Deve essere vero.",
    "Neppure io.",
    "Ciao Elena!",
    "Ehi Steve!",
    "Ciao Eve.",
    "Nev, ascolta.",
    "Breve e intenso.",
    "Le navi partono.",
    "Ehi, Nina!",
    "Ciao Vera!",
    "Un sorso, poi beve ancora.",
    "Nevermind.",
]
NEAR_WORDS = [
    "Deve.",
    "Beve.",
    "Nave!",
    "Breve.",
    "Eva!",
    "Nevio!",
    "Nevica.",
    "Neo.",
    "Lieve.",
    "Nervi.",
    "Ehi Eva!",
    "Ciao Eva.",
    "Ehi!",
    "Ciao!",
    "Nello!",
    "Vede.",
    "Neri!",
    "Nina!",
]
EN_NEG = [
    "I have never been there.",
    "Nevada is a big state.",
    "Heaven can wait.",
    "Hey Eve, come here.",
    "The navy ship sailed.",
    "Hey Steve!",
    "Never mind.",
    "Nevertheless, we go.",
    "Hey, never again.",
    "Chow time, everybody.",
    "Hey Nate.",
    "Neville is late.",
    "Hi Neva!",
    "Seven eleven.",
    "Hey there!",
    "Nobody knew.",
    "Navy blue.",
    "Hey, Nadia.",
    "Evening news.",
    "Every day.",
]


def split(items, k):
    """Deterministic split by index: 0 -> 'test', 1 -> 'val', else 'train'."""
    want = {"test": lambda m: m == 0, "val": lambda m: m == 1, "train": lambda m: m >= 2}[k]
    return [x for i, x in enumerate(items) if want(i % 5)]


def expand_it(seed=0):
    """Extra Italian neve sentences from templates (more contexts around the word)."""
    rnd = random.Random(seed)
    pre = [
        "Oggi c'è",
        "Domani ci sarà",
        "Ho visto",
        "Mi piace",
        "Non sopporto",
        "Guarda quanta",
        "Ecco la",
        "Arriva la",
        "Parlano della",
        "Speriamo nella",
        "Basta con la",
        "Senti la",
    ]
    post = [
        "",
        " in montagna",
        " sul tetto",
        " fuori dalla finestra",
        " a Milano",
        " di stanotte",
        " fresca",
        " ghiacciata",
        " sul balcone",
        " del mese scorso",
    ]
    out = [f"{a} neve{b}." for a, b in itertools.product(pre, post)]
    rnd.shuffle(out)
    return out


def splits():
    """Return dict split -> list of hard-negative texts (Italian)."""
    neve = NEVE_SENT + expand_it()
    res = {}
    for k in ("train", "val", "test"):
        res[k] = {
            "neve": split(neve, k),
            "near": split(NEAR_SENT, k) + split(NEAR_WORDS, k),
            "en": split(EN_NEG, k),
        }
    return res


if __name__ == "__main__":
    s = splits()
    for k, v in s.items():
        print(k, {a: len(b) for a, b in v.items()})
