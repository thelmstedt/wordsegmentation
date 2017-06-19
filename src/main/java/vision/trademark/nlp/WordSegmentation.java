package vision.trademark.nlp;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
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


    public List<String> segment(String text) {
        text = text == null ? "" : text.toLowerCase().trim().replaceAll("'", "");

        List<Pair<Position, Double>> meaningfulWords = meaningfulWords(minLength, text, ngramTree);

        List<Set<Pair<Position, Double>>> sets = connectedSets(meaningfulWords);

        List<Triple<Integer, Integer, List<String>>> postComponents = new ArrayList<>();
        for (Set<Pair<Position, Double>> each : sets) {
            Triple<Integer, Integer, List<String>> pairs = optComponent(each);
            postComponents.add(pairs);
        }

        Set<Integer> meaningfulIndices = new HashSet<>();
        for (Triple<Integer, Integer, List<String>> x : postComponents) {
            meaningfulIndices.addAll(IntStream.range(x.getLeft(), x.getMiddle() + 1).boxed().collect(Collectors.toList()));
        }

        List<Pair<Integer, Integer>> nonMeaningfulRanges = missingSegments(text, meaningfulIndices);

        List<Pair<Integer, Integer>> overallPosList = new ArrayList<>();
        overallPosList.addAll(nonMeaningfulRanges);

        Map<Pair<Integer, Integer>, List<String>> meaningfulDic = new HashMap<>();
        for (Triple<Integer, Integer, List<String>> x : postComponents) {
            Pair<Integer, Integer> p = Pair.of(x.getLeft(), x.getMiddle());
            overallPosList.add(p);
            meaningfulDic.put(p, x.getRight());
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

    private List<Pair<Position, Double>> meaningfulWords(int minLength, String text, Map<String, List<String>> ngramTree) {
        List<WordPosition> cuts = cuts(minLength, text);
        List<WordPosition> candidateList = new ArrayList<>();
        for (WordPosition cut : cuts) {
            if (ngramTree.containsKey(cut.prefix.ngram)) {
                candidateList.add(cut);
            }
        }
        Collections.reverse(candidateList);

        List<Pair<Position, Double>> meaningfulWords = new ArrayList<>();
        for (WordPosition x : candidateList) {
            int start = x.prefix.start;
            String recovered = x.prefix.ngram + x.suffix;
            char c = x.prefix.ngram.charAt(0);
            if ('a' == c) {
                meaningfulWords.add(Pair.of(new Position("a", start, start + "a".length() - 1), getUnigramScore("a")));
            }

            for (String word : ngramTree.get(x.prefix.ngram)) {
                if (recovered.contains(word)) {
                    if (text.substring(start, start + word.length()).equals(word)) {
                        meaningfulWords.add(Pair.of(
                                new Position(word, start, start + word.length() - 1), getUnigramScore(word)
                        ));
                    }
                }
            }
        }
        meaningfulWords.sort(Comparator.comparingInt(it -> it.getLeft().start));
        return meaningfulWords;
    }

    /**
     * Places {@param meaningfulWords} in a graph (connected by positions), returning all connected subgraphs as sets.
     */
    private List<Set<Pair<Position, Double>>> connectedSets(List<Pair<Position, Double>> meaningfulWords) {
        DirectedGraph<Pair<Position, Double>, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);
        for (Pair<Position, Double> x : meaningfulWords) {
            graph.addVertex(x);
        }
        Set<Set<Position>> seen = new HashSet<>();
        for (Pair<Position, Double> x : meaningfulWords) {
            for (Pair<Position, Double> y : meaningfulWords) {
                if (!x.equals(y)) {
                    if (!isNotIntersecting(x.getLeft(), y.getLeft())) {
                        Set<Position> check = new HashSet<>();
                        check.add(x.getLeft());
                        check.add(y.getLeft());
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

    /**
     * Maps ngrams of length {@param minLength} to the words starting with them.
     * Note we need to calculate this for the specific length we're allowing,
     * so we either do this for each segmentation (and incur the CPU cost, but
     * allow changing the minLength) or at construction (and incur a memory cost,
     * and restrict it to a single length).
     *
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

    private Triple<Integer, Integer, List<String>> optComponent(Set<Pair<Position, Double>> in) {
        List<Pair<Position, Double>> meaningfulWords = in.stream()
                .sorted(Comparator.comparingInt(x -> x.getLeft().end))
                .collect(Collectors.toList());

        List<Pair<Pair<Position, Double>, MaybeRange>> lst = new CircularList<>();
        for (Pair<Position, Double> tu1 : meaningfulWords) {
            int pos = 0;
            ArrayList<Integer> prevList = new ArrayList<>();
            for (Pair<Position, Double> tu2 : meaningfulWords) {
                Position p1 = tu1.getLeft();
                Position p2 = tu2.getLeft();
                if (isNotIntersecting(p1, p2) && p1.start == p2.end + 1) {
                    prevList.add(pos + 1);
                }
                pos++;
            }
            if (!prevList.isEmpty()) {
                Collections.reverse(prevList);
            }
            if (prevList.isEmpty()) {
                lst.add(Pair.of(tu1, new MaybeRange(0, Optional.empty())));
            } else if (prevList.size() == 1) {
                lst.add(Pair.of(tu1, new MaybeRange(prevList.get(0), Optional.empty())));
            } else {
                lst.add(Pair.of(tu1, new MaybeRange(prevList.get(0), Optional.of(prevList.get(1)))));
            }
        }

        List<Position> path = calculatePaths(lst);

        List<String> wordList = new ArrayList<>();
        for (Position position : path) {
            wordList.add(position.ngram);
        }

        int s = path.get(0).start;
        int e = lst.get(lst.size() - 1).getLeft().getLeft().end;
        return Triple.of(s, e, wordList);

    }

    private List<Position> calculatePaths(List<Pair<Pair<Position, Double>, MaybeRange>> lst) {
        int j = lst.size();

        HashMap<Integer, Double> memo = new HashMap<>();
        Double result = opt(j, lst, memo);

        List<Integer> maxV =
                memo.entrySet().stream()
                        .sorted(Comparator.comparing(Map.Entry::getValue))
                        .filter(x -> x.getValue().equals(result))
                        .map(Map.Entry::getKey).collect(Collectors.toList());
        j = maxV.get(maxV.size() - 1) + 1;

        List<Position> path = new ArrayList<>();
        path(j, path, memo, lst);
        Collections.reverse(path);
        return path;
    }

    private void path(int j,
                      List<Position> path,
                      HashMap<Integer, Double> memo,
                      List<Pair<Pair<Position, Double>, MaybeRange>> lst) {
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
                path.add(lst.get(0).getLeft().getLeft());
            }
        } else {
            Pair<Pair<Position, Double>, MaybeRange> prev = lst.get(j - 1);
            MaybeRange range = prev.getRight();
            int tempI = 0;
            if (!range.end.isPresent()) {
                path.add(prev.getLeft().getLeft());
                path(prev.getRight().get(tempI), path, memo, lst);
            } else {
                List<Double> pList = Arrays.asList(memo.get(range.start - 1), memo.get(range.end.get() - 1));
                Double maxP = Double.max(pList.get(0), pList.get(1));
                path.add(prev.getLeft().getLeft());
                path(pList.get(0).equals(maxP) ? range.start : range.end.get(), path, memo, lst);

            }
        }
    }

    private Double opt(int j, List<Pair<Pair<Position, Double>, MaybeRange>> lst, HashMap<Integer, Double> memo) {
        if (j == 0) return null;

        if (memo.containsKey(j - 1)) {
            return memo.get(j - 1);
        } else {
            Pair<Pair<Position, Double>, MaybeRange> next = lst.get(j - 1);
            Double max = max(
                    //choose j
                    add(opt(next.getRight().start, lst, memo),
                            next.getLeft().getRight(),
                            penalize(j, next.getRight().start, lst)),
                    //not choose j and jump to j-1 only when nesrest overlpping word has the same finish position
                    lst.get(j - 2).getLeft().getLeft().end == next.getLeft().getLeft().end ? opt(j - 1, lst, memo) : null
            );
            memo.put(j - 1, max);
            return max;
        }

    }

    private Double max(Double d1, Double d2) {
        if (d1 == null) return d2;
        if (d2 == null) return d1;
        return Double.max(d1, d2);
    }

    private Double add(Double d1, Double d2, Double d3) {
        if (d1 == null) {
            return d2 + d3;
        } else {
            return d1 + d2 + d3;
        }
    }

    private Double penalize(int current, Integer prev, List<Pair<Pair<Position, Double>, MaybeRange>> lst) {
        double penalty = -10;
        double bigramReward = 0;
        double gap;
        Pair<Pair<Position, Double>, MaybeRange> curr = lst.get(current - 1);
        Pair<Pair<Position, Double>, MaybeRange> previous = lst.get(prev - 1);

        if (prev == 0) {
            gap = penalty * (curr.getLeft().getLeft().start);
        } else if (curr.getLeft().getLeft().start - previous.getLeft().getLeft().end == 1) {
            String bigram = String.format("%s %s", previous.getLeft().getLeft().ngram, curr.getLeft().getLeft().ngram);
            if (bigramCounts.containsKey(bigram)) {
                bigramReward = (bigramCounts.get(bigram) / 1024908267229.0 / previous.getLeft().getRight()) - curr.getLeft().getRight();
            }
            gap = 0;
        } else {
            gap = 0;
        }
        return gap + bigramReward;
    }


    private List<WordPosition> cuts(int minLen, String word) {
        ArrayList<WordPosition> xs = new ArrayList<>();
        int counter = 0;
        for (int i = minLen; i < word.length() + 1; i++) {
            xs.add(new WordPosition(word.substring(counter, i), counter, counter + minLen, word.substring(i, word.length())));
            counter++;
        }
        return xs;
    }

    private boolean isNotIntersecting(Position left, Position right) {
        return right.end < left.start || right.start > left.end;
    }

    private double getUnigramScore(String word) {
        double scale = Math.log10(1024908267229.0);
        Long x = fullUnigramCounts.get(word);
        return x != null
                ? Math.log10(x) - scale
                : Math.log10(10.0) - (scale + Math.log10(Math.pow(10, word.length())));
    }

    // Simplifying porting from python
    private class CircularList<E> extends ArrayList<E> {
        @Override
        public E get(int i) {
            return super.get(i < 0 ? i + size() : i);
        }

    }


    private class MaybeRange {
        final Integer start;
        final Optional<Integer> end;

        MaybeRange(Integer start, Optional<Integer> end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return new ToStringBuilder(this, ToStringStyle.NO_CLASS_NAME_STYLE)
                    .append("start", start)
                    .append("end", end)
                    .toString();
        }

        Integer get(int tempI) {
            if (tempI == 0) {
                return start;
            } else {
                return end.get();
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            MaybeRange maybeRange = (MaybeRange) o;
            return Objects.equals(start, maybeRange.start) &&
                    Objects.equals(end, maybeRange.end);
        }

        @Override
        public int hashCode() {
            return Objects.hash(start, end);
        }
    }


    private static class Position {
        private final String ngram;
        private final int start;

        private final int end;

        Position(String ngram, int start, int end) {
            this.ngram = ngram;
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return new ToStringBuilder(this, ToStringStyle.NO_CLASS_NAME_STYLE)
                    .append("ngram", ngram)
                    .append("start", start)
                    .append("end", end)
                    .toString();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Position)) return false;
            Position position = (Position) o;
            return start == position.start &&
                    end == position.end &&
                    Objects.equals(ngram, position.ngram);
        }

        @Override
        public int hashCode() {
            return Objects.hash(ngram, start, end);
        }

    }

    private static class WordPosition {
        private final Position prefix;

        private final String suffix;

        WordPosition(String ngram, int start, int end, String suffix) {
            this.prefix = new Position(ngram, start, end);
            this.suffix = suffix;
        }

        @Override
        public String toString() {
            return new ToStringBuilder(this, ToStringStyle.NO_CLASS_NAME_STYLE)
                    .append("prefix", prefix)
                    .append("suffix", suffix)
                    .toString();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof WordPosition)) return false;
            WordPosition that = (WordPosition) o;
            return Objects.equals(prefix, that.prefix) &&
                    Objects.equals(suffix, that.suffix);
        }

        @Override
        public int hashCode() {
            return Objects.hash(prefix, suffix);
        }

    }


}
