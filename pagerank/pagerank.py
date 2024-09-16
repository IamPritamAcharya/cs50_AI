
import os
import random
import re
import sys

# Damping Factor - probablity that a link is selected from the current page. Otherwise a page from the corpus is switched to at random.
DAMPING = 0.85
SAMPLES = 10000


def main():
    """ Main function to run pagerank algorithm """
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    If a page has no outgoing links, returns an equal probability for all pages in the corpus
    """
    prob_dist = {page_name: 0 for page_name in corpus}

    if len(corpus[page]) == 0:
        # If the page has no outgoing links, choose randomly among all pages
        for page_name in prob_dist:
            prob_dist[page_name] = 1 / len(corpus)
    else:
        # Probability of choosing a link from the current page
        link_prob = damping_factor / len(corpus[page])
        # Probability of choosing a random page (teleportation)
        random_prob = (1 - damping_factor) / len(corpus)

        for page_name in prob_dist:
            if page_name in corpus[page]:
                prob_dist[page_name] += link_prob
            prob_dist[page_name] += random_prob

    return prob_dist



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_rank[current_page] += 1
        prob_dist = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(prob_dist.keys()), weights=prob_dist.values(), k=1)[0]

    # Normalize the page rank values
    page_rank = {page: rank / n for page, rank in page_rank.items()}
    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_page_rank = {page: 0 for page in corpus}

    while True:
        max_rank_change = 0

        for page in corpus:
            new_page_rank[page] = (1 - damping_factor) / num_pages

            for other_page, links in corpus.items():
                if page in links and len(links) > 0:
                    new_page_rank[page] += damping_factor * (page_rank[other_page] / len(links))
                elif len(links) == 0:
                    new_page_rank[page] += damping_factor * (page_rank[other_page] / num_pages)

        for page in corpus:
            rank_change = abs(new_page_rank[page] - page_rank[page])
            if rank_change > max_rank_change:
                max_rank_change = rank_change

        if max_rank_change < 0.001:
            break
        page_rank = new_page_rank.copy()

    return page_rank



if __name__ == "__main__":
    main()