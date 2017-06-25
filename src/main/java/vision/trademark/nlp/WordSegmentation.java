package vision.trademark.nlp;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.jgrapht.DirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public class WordSegmentation {
    private static final double TOTAL = 1024908267229.0;

    private final Map<String, Long> bigramCounts;
    private final Map<String, Long> fullUnigramCounts;
    private final Map<String, List<String>> ngramTree;
    private final int minLength;

    /**
     * To avoid paying the cost of calculating ngramTree for each segmentation,
     * we only allow a single minLength for all segments.
     */
    public WordSegmentation(int minLength) {
        this.bigramCounts = loadWordList("/bigrams.txt.gz");
        this.fullUnigramCounts = loadWordList("/unigrams.txt.original.gz");

        this.ngramTree = ngramTree(minLength, loadWordList("/unigrams.txt.gz"));
        this.minLength = minLength;
    }


    public List<String> segment(String text) {
        text = text == null ? "" : text.toLowerCase().trim().replaceAll("'", "");

        List<ScorePosition<String>> meaningfulWords = meaningfulWords(minLength, text, ngramTree);

        List<Set<ScorePosition<String>>> sets = connectedSets(meaningfulWords);

        List<Position<List<String>>> postComponents = new ArrayList<>();
        for (Set<ScorePosition<String>> x : sets) {
            postComponents.add(optComponent(x));
        }

        Set<Integer> meaningfulIndices = new HashSet<>();
        for (Position<List<String>> x : postComponents) {
            meaningfulIndices.addAll(IntStream.range(x.getStart(), x.getEnd() + 1).boxed().collect(Collectors.toList()));
        }

        List<Pair<Integer, Integer>> nonMeaningfulRanges = missingSegments(text, meaningfulIndices);

        List<Pair<Integer, Integer>> overallPosList = new ArrayList<>();
        overallPosList.addAll(nonMeaningfulRanges);

        Map<Pair<Integer, Integer>, List<String>> meaningfulDic = new HashMap<>();
        for (Position<List<String>> x : postComponents) {
            Pair<Integer, Integer> p = Pair.of(x.getStart(), x.getEnd());
            overallPosList.add(p);
            meaningfulDic.put(p, x.getNgram());
        }
        overallPosList.sort(Comparator.comparing(Pair::getLeft));

        List<String> returnList = new ArrayList<>();
        for (Pair<Integer, Integer> x : overallPosList) {
            if (meaningfulDic.containsKey(x)) {
                returnList.addAll(meaningfulDic.get(x));
            } else {
                Integer left = x.getLeft();
                Integer right = x.getRight();
                String substring = text.substring(left, right + 1);
                returnList.add(substring);
            }
        }
        return returnList;
    }

    private List<ScorePosition<String>> meaningfulWords(int minLength, String text, Map<String, List<String>> ngramTree) {
        List<ScorePosition<String>> meaningfulWords = new ArrayList<>();

        List<SuffixedPosition> cuts = cuts(minLength, text);
        for (int i = cuts.size() - 1; i >= 0; i--) {
            SuffixedPosition x = cuts.get(i);
            List<String> words = ngramTree.get(x.getPrefix());
            if (words != null) {
                int start = x.getStart();
                String recovered = x.getPrefix() + x.getSuffix();
                char c = x.getPrefix().charAt(0);
                if ('a' == c) {
                    meaningfulWords.add(new ScorePosition<>("a", getUnigramScore("a"), start, start + "a".length() - 1));
                }

                for (String word : words) {
                    if (recovered.contains(word)) {
                        if (text.substring(start, start + word.length()).equals(word)) {
                            meaningfulWords.add(new ScorePosition<>(word, getUnigramScore(word), start, start + word.length() - 1));
                        }
                    }
                }
            }
        }
        meaningfulWords.sort(Comparator.comparingInt(Position::getStart));
        return meaningfulWords;
    }

    /**
     * Places {@param meaningfulWords} in a graph (connected by positions), returning all connected subgraphs as sets.
     *
     * @param meaningfulWords
     */
    private List<Set<ScorePosition<String>>> connectedSets(List<ScorePosition<String>> meaningfulWords) {
        DirectedGraph<ScorePosition<String>, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);
        for (ScorePosition<String> x : meaningfulWords) {
            graph.addVertex(x);
        }
        Set<Set<Position>> seen = new HashSet<>();
        for (ScorePosition x : meaningfulWords) {
            for (ScorePosition y : meaningfulWords) {
                if (!x.equals(y)) {
                    if (!isNotIntersecting(x, y)) {
                        Set<Position> check = new HashSet<>();
                        check.add(x);
                        check.add(y);
                        if (!seen.contains(check)) {
                            graph.addEdge(x, y);
                            seen.add(check);
                        }
                    }
                }
            }
        }
        return new ConnectivityInspector<>(graph).connectedSets();
    }

    private List<Pair<Integer, Integer>> missingSegments(String text, Set<Integer> meaningfulIndices) {
        ArrayList<Pair<Integer, Integer>> pairs = new ArrayList<>();

        List<Integer> current = new ArrayList<>();
        for (int i = 0; i < text.length(); i++) {
            if (!meaningfulIndices.contains(i)) {
                if (current.isEmpty()) {
                    current.add(i);
                } else {
                    if (current.get(current.size() - 1) == i - 1) {
                        current.add(i);
                    } else {
                        pairs.add(Pair.of(current.get(0), current.get(current.size() - 1)));
                        current = new ArrayList<>();
                        current.add(i);
                    }
                }
            }
        }
        if (!current.isEmpty()) {
            pairs.add(Pair.of(current.get(0), current.get(current.size() - 1)));
        }
        return pairs;
    }

    private ScorePosition<List<String>> optComponent(Set<ScorePosition<String>> in) {
        List<ScorePosition<String>> meaningfulWords = in.stream()
                .sorted(Comparator.comparing(Position::getEnd))
                .collect(Collectors.toList());

        List<Pair<ScorePosition<String>, MaybeRange>> lst = new CircularList<>();
        for (ScorePosition<String> p1 : meaningfulWords) {
            int pos = 0;
            ArrayList<Integer> prevList = new ArrayList<>();
            for (ScorePosition<String> p2 : meaningfulWords) {
                if (isNotIntersecting(p1, p2) && p1.getStart() == p2.getEnd() + 1) {
                    prevList.add(pos + 1);
                }
                pos++;
            }
            if (!prevList.isEmpty()) {
                Collections.reverse(prevList);
            }
            if (prevList.isEmpty()) {
                lst.add(Pair.of(p1, new MaybeRange(0, Optional.empty())));
            } else if (prevList.size() == 1) {
                lst.add(Pair.of(p1, new MaybeRange(prevList.get(0), Optional.empty())));
            } else {
                lst.add(Pair.of(p1, new MaybeRange(prevList.get(0), Optional.of(prevList.get(1)))));
            }
        }

        Pair<List<Position<String>>, Double> path = calculatePaths(lst);

        List<String> wordList = new ArrayList<>();
        for (Position<String> position : path.getLeft()) {
            wordList.add(position.getNgram());
        }

        int s = path.getLeft().get(0).getStart();
        int e = lst.get(lst.size() - 1).getLeft().getEnd();
        return new ScorePosition<>(wordList, path.getRight(), s, e);

    }

    private Pair<List<Position<String>>, Double> calculatePaths(List<Pair<ScorePosition<String>, MaybeRange>> lst) {
        int j = lst.size();

        HashMap<Integer, Double> memo = new HashMap<>();
        Double result = opt(j, lst, memo);

        List<Integer> maxV =
                memo.entrySet().stream()
                        .sorted(Comparator.comparing(Map.Entry::getValue))
                        .filter(x -> x.getValue().equals(result))
                        .map(Map.Entry::getKey).collect(Collectors.toList());
        j = maxV.get(maxV.size() - 1) + 1;

        List<Position<String>> path = new ArrayList<>();
        path(j, path, memo, lst);
        Collections.reverse(path);
        return Pair.of(path, result);
    }

    private void path(int j,
                      List<Position<String>> path,
                      HashMap<Integer, Double> memo,
                      List<Pair<ScorePosition<String>, MaybeRange>> lst) {
        if (j == 0) {
            return;
        }

        boolean b = j - 2 >= 0
                ? Objects.equals(memo.get(j - 1), memo.get(j - 2))
                : memo.get(0) != null && memo.get(0).equals(0d);
        if (b) {
            if (j != 1) {
                path(j - 1, path, memo, lst);
            } else {
                path.add(lst.get(0).getLeft());
            }
        } else {
            Pair<ScorePosition<String>, MaybeRange> prev = lst.get(j - 1);
            MaybeRange range = prev.getRight();
            if (!range.end.isPresent()) {
                path.add(prev.getLeft());
                path(prev.getRight().start, path, memo, lst);
            } else {
                List<Double> pList = Arrays.asList(memo.get(range.start - 1), memo.get(range.end.get() - 1));
                Double maxP = Double.max(pList.get(0), pList.get(1));
                path.add(prev.getLeft());
                path(pList.get(0).equals(maxP) ? range.start : range.end.get(), path, memo, lst);

            }
        }
    }

    private Double opt(int j, List<Pair<ScorePosition<String>, MaybeRange>> lst, HashMap<Integer, Double> memo) {
        if (j == 0) return null;

        if (memo.containsKey(j - 1)) {
            return memo.get(j - 1);
        } else {
            Pair<ScorePosition<String>, MaybeRange> next = lst.get(j - 1);
            Double max = max(
                    //choose j
                    add(opt(next.getRight().start, lst, memo),
                            next.getLeft().getScore(),
                            penalize(j, next.getRight().start, lst)),
                    //not choose j and jump to j-1 only when nesrest overlpping word has the same finish position
                    lst.get(j - 2).getLeft().getEnd() == next.getLeft().getEnd() ? opt(j - 1, lst, memo) : null
            );
            memo.put(j - 1, max);
            return max;
        }
    }

    private Double penalize(int curri, int previ, List<Pair<ScorePosition<String>, MaybeRange>> lst) {
        double penalty = -10.0; //TODO why this?
        Pair<ScorePosition<String>, MaybeRange> curr = lst.get(curri - 1);
        Pair<ScorePosition<String>, MaybeRange> prev = lst.get(previ - 1);

        if (previ == 0) { // penalize gaps between words
            return penalty * (curr.getLeft().getStart());
        }

        return stupidBackoff(curr, prev);
    }

    /**
     * Stupid backoff from http://www.aclweb.org/anthology/D07-1090.pdf
     */
    private Double stupidBackoff(Pair<ScorePosition<String>, MaybeRange> curr, Pair<ScorePosition<String>, MaybeRange> prev) {
        double alpha = 0.4;
        if (curr.getLeft().getStart() - prev.getLeft().getEnd() == 1) {
            Long count = bigramCounts.get(String.format("%s %s", prev.getLeft().getNgram(), curr.getLeft().getNgram()));
            if (count != null) {
                return (count / TOTAL) / prev.getLeft().getScore();
            }
        }
        return prev.getLeft().getScore() * alpha;
    }

    private List<SuffixedPosition> cuts(int minLen, String word) {
        ArrayList<SuffixedPosition> xs = new ArrayList<>();
        int counter = 0;
        for (int i = minLen; i < word.length() + 1; i++) {
            xs.add(new SuffixedPosition(word.substring(counter, i), counter, counter + minLen, word.substring(i, word.length())));
            counter++;
        }
        return xs;
    }

    private double getUnigramScore(String word) {
        double scale = Math.log10(TOTAL);
        Long x = fullUnigramCounts.get(word);
        return x != null
                ? Math.log10(x) - scale
                : Math.log10(10.0) - (scale + Math.log10(Math.pow(10, word.length())));
    }

    private static boolean isNotIntersecting(Position<String> left, Position<String> right) {
        return right.getEnd() < left.getStart() || right.getStart() > left.getEnd();
    }

    private static Double max(Double d1, Double d2) {
        if (d1 == null) return d2;
        if (d2 == null) return d1;
        return Double.max(d1, d2);
    }

    private static Double add(Double d1, Double d2, Double d3) {
        if (d1 == null) {
            return d2 + d3;
        } else {
            return d1 + d2 + d3;
        }
    }


    // Simplifying porting from python
    private class CircularList<E> extends ArrayList<E> {

        @Override
        public E get(int i) {
            return super.get(i < 0 ? i + size() : i);
        }

    }

    /**
     * Maps ngrams of length {@param minLength} to the words starting with them.
     * Note we need to calculate this for the specific length we're allowing,
     * so we either do this for each segmentation (and incur the CPU cost, but
     * allow changing the minLength) or at construction (and incur a memory cost,
     * and restrict it to a single length).
     * <p>
     * We've opted to the latter.
     */
    private Map<String, List<String>> ngramTree(int minLength, Map<String, Long> unigramCounts) {
        Map<String, List<String>> ngramTree = new HashMap<>();
        for (Map.Entry<String, Long> e : unigramCounts.entrySet()) {
            String entry = e.getKey();
            if (entry.length() >= minLength) {
                String cut = entry.substring(0, minLength);

                if (!ngramTree.containsKey(cut)) {
                    ArrayList<String> v = new ArrayList<>();
                    v.add(entry);
                    ngramTree.put(cut, v);
                } else {
                    ngramTree.get(cut).add(entry);
                }
            }
        }

        return ngramTree;
    }

    private static Map<String, Long> loadWordList(String resourcePath) {
        try (InputStream in = new GZIPInputStream(WordSegmentation.class.getResourceAsStream(resourcePath));
             BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            HashMap<String, Long> xs = new HashMap<>();
            r.lines()
                    .filter(StringUtils::isNotBlank)
                    .map(x -> x.split("\t"))
                    .filter(x -> x.length == 2)
                    .forEach(x -> {
                        String key = x[0].trim().toLowerCase();
                        Long val = Long.parseLong(x[1]);
                        xs.put(key, val);
                    });
            return xs;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
