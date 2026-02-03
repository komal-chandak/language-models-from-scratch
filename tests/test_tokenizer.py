import pytest
import tiktoken
import os
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

# -------------------------------------------------
# test data

test_strings = ["","hello world","hello world!!!? (안녕하세요!) lol123 😉", "FILE:taylorswift.txt"]

specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()

llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

test_text = """
<|endoftext|>Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Her versatile artistry, songwriting, and entrepreneurship have influenced the music industry, popular culture, and politics, and her life is a subject of widespread media coverage.
Swift began professional songwriting at 14 and signed with Big Machine Records in 2005 to become a country singer. She released six studio albums under the label, four of them to country radio, starting with Taylor Swift (2006). Her next, Fearless (2008), explored country pop, and its singles "Love Story" and "You Belong with Me" catapulted her to mainstream fame. Speak Now (2010) infused rock influences, while Red (2012) experimented with electronic elements and featured Swift's first Billboard Hot 100 number-one song, "We Are Never Ever Getting Back Together". She departed from her country image with 1989 (2014), a synth-pop album supported by the chart-topping songs "Shake It Off", "Blank Space", and "Bad Blood". Media scrutiny inspired the hip-hop-influenced Reputation (2017) and its number-one single "Look What You Made Me Do".
After signing with Republic Records in 2018, Swift released the eclectic pop album Lover (2019) and the autobiographical documentary Miss Americana (2020). She explored indie folk styles on the 2020 albums Folklore and Evermore, subdued electropop on Midnights (2022), and re-recorded four albums subtitled Taylor's Version after a dispute with Big Machine. These albums spawned the number-one songs "Cruel Summer", "Cardigan", "Willow", "Anti-Hero", "All Too Well", and "Is It Over Now?". Her Eras Tour (2023–2024) and its accompanying concert film became the highest-grossing tour and concert film of all time, respectively. Swift has directed several music videos and films such as Folklore: The Long Pond Studio Sessions (2020) and All Too Well: The Short Film (2021).
Discography
            Filmography
Tours
See also
Footnotes
References
Toggle References           subsection
External links
Taylor Swift




136 languages
Article
                    Talk
Read
View source



View history
One of the world's best-selling musicians, with over 200 million records sold as of 2019, Swift has been named Global Recording Artist of the Year three times by the International <|fim_suffix|> Federation of the Phonographic Industry, whereas six of her albums have opened <|fim_middle|> with over a million sales in a week. She is the highest-grossing female touring act, the most-streamed woman on Spotify and Apple Music, and the first billionaire with music as the main source of income. The 2023 Time Person of the Year, Swift has appeared on lists such as Rolling Stone's 100 Greatest Songwriters of All Time, Billboard's Greatest of All Time Artists, and Forbes' World's 100 Most Powerful Women. Her accolades include 14 Grammy Awards (featuring a record four Album of the Year wins), a Primetime Emmy Award, 40 American Music Awards, 40 Billboard Music Awards, and 23 MTV Video Music Awards.
Life and career<|endofprompt|>""".strip()

special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def unpack(text):
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(dirname, text[5:])
        with open(filename, 'r' ) as file:
            data = file.read()
        return data
    else:
        return text

# ---------------------       tests        ----------------------------

# test encode/decode identity for a few different strings
@pytest.mark.parametrize("text", test_strings)
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer, GPT4Tokenizer])
def test_encode_decode_identity(text, tokenizer_factory):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    assert text==decoded_text
 
# test that our tokenizer matches the official GPT-4 tokenizer
@pytest.mark.parametrize("text", test_strings)
def test_gpt4_tiktoken_equality(text):
    text = unpack(text)
    tokenizer = GPT4Tokenizer()
    gpt4_tokenizer_ids = tokenizer.encode(text)
    enc = tiktoken.get_encoding('cl100k_base')
    tiktoken_ids = enc.encode(text)
    assert gpt4_tokenizer_ids == tiktoken_ids

# test the handling of special tokens
def test_gpt4_tiktoken_equality_special_tokens():
    tokenizer = GPT4Tokenizer()
    enc = tiktoken.get_encoding('cl100k_base')
    gpt4_tokenizer_ids = tokenizer.encode(specials_string, allowed_special="all")
    tiktoken_ids = enc.encode(specials_string, allowed_special="all")
    assert gpt4_tokenizer_ids == tiktoken_ids

def test_save_load():
    text = llama_text
    tokenizer = RegexTokenizer()
    tokenizer.train(text, 256+64)
    tokenizer.register_special_tokens(special_tokens)
    # verify that decode(encode(x)) == x
    assert tokenizer.decode(tokenizer.encode(text, "all"))==text

    # verify save/load is working
    ids = tokenizer.encode(text, "all")
    
    tokenizer.save('test_tokenizer_tmp')  # save
    # reload
    tokenizer = RegexTokenizer()
    tokenizer.load('test_tokenizer_tmp.model')  
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    assert tokenizer.encode(text, "all") == ids

    for file in ["test_tokenizer_tmp.model", "test_tokenizer_tmp.vocab"]:
        os.remove(file)

def test_compression_basic_vs_regex():
    text = llama_text
    raw_bytes = len(text.encode("utf-8"))
    test_raw_bytes = len(test_text.encode("utf-8"))

    # Basic tokenizer
    basic = BasicTokenizer()
    basic.train(text, 256 + 64)
    ids = basic.encode(text)
    basic_ratio = raw_bytes / len(ids)
    basic_ratio_test = test_raw_bytes / len(basic.encode(test_text))

    # Regex tokenizer
    regex = RegexTokenizer()
    regex.train(text, 256 + 64)
    ids = regex.encode(text, "all")
    regex_ratio = raw_bytes / len(ids)
    regex_ratio_test = test_raw_bytes / len(regex.encode(test_text, "all"))

    # GPT4 encoder
    gpt4 = GPT4Tokenizer()
    ids = gpt4.encode(text, "all")
    gpt4_ratio = raw_bytes / len(ids)
    gpt4_ratio_test = test_raw_bytes / len(gpt4.encode(test_text, "all"))

    print(f"\nBasicTokenizer compression: {basic_ratio:.2f}x")
    print(f"RegexTokenizer compression: {regex_ratio:.2f}x")
    print(f"GPT4Tokenizer compression: {gpt4_ratio:.2f}x")
    print("\nTest string:")
    print(f"BasicTokenizer compression: {basic_ratio_test:.2f}x")
    print(f"RegexTokenizer compression: {regex_ratio_test:.2f}x")
    print(f"GPT4Tokenizer compression: {gpt4_ratio_test:.2f}x")

if __name__ == '__main__':
    pytest.main()
