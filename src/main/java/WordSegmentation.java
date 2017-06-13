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
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public class WordSegmentation {
    private final Map<String, Long> bigramCounts;
    private final Map<String, Long> unigramCounts;
    private final Map<String, Long> fullUnigramCounts;

    /**
     * Users are expected to construct this once and keep it around.
     */
    public WordSegmentation() {
        bigramCounts = loadWordList("/bigrams.txt.gz");
        unigramCounts = loadWordList("/unigrams.txt.gz");
        fullUnigramCounts = loadWordList("/unigrams.txt.original.gz"); //just for scoring. TODO necessary?
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


    public List<String> segment(int minLength, String text) {
        text = text == null ? "" : text.toLowerCase().trim().replaceAll("'", "");

        Pair<Map<String, Long>, Map<String, List<String>>> ngrams = calculateNgrams(minLength);
        Map<String, Long> ngramDistribution = ngrams.getLeft();
        Map<String, List<String>> ngramTree = ngrams.getRight();

        List<WordPosition> cuts = cuts(minLength, text);
        Map<Position, String> pairDic = new HashMap<>();
        List<Pair<Long, Position>> candidateList = new ArrayList<>();
        for (WordPosition cut : cuts) {
            pairDic.put(cut.prefix, cut.suffix);
            Long aLong = ngramDistribution.get(cut.prefix.ngram);
            if (aLong != null) {
                candidateList.add(Pair.of(aLong, cut.prefix));
            }
        }
        Collections.reverse(candidateList);

        List<Pair<Position, Double>> meaningfulWords = new ArrayList<>();
        for (Pair<Long, Position> x : candidateList) {
            Position prefixPart = x.getRight();
            int start = prefixPart.start;
            char c = prefixPart.ngram.charAt(0);
            if ('a' == c) {
                meaningfulWords.add(Pair.of(new Position("a", start, start + "a".length() - 1), getUnigramScore("a")));
            }

            for (String word : ngramTree.get(prefixPart.ngram)) {
                if ((prefixPart.ngram + pairDic.get(prefixPart)).contains(word)) {
                    if (text.substring(start, start + word.length()).equals(word)) {
                        meaningfulWords.add(Pair.of(
                                new Position(word, start, start + word.length() - 1), getUnigramScore(word)
                        ));
                    }
                }
            }
        }
        meaningfulWords.sort(Comparator.comparingInt(it -> it.getLeft().start));

        List<Set<Pair<Position, Double>>> sets = connectedSets(meaningfulWords);

        List<Triple<Integer, Integer, List<String>>> postComponents = new ArrayList<>();
        for (Set<Pair<Position, Double>> each : sets) {
            Triple<Integer, Integer, List<String>> pairs = optComponent(each);
            postComponents.add(pairs);
        }

        List<Pair<Integer, Integer>> nonMeaningfulRanges = nonMeaningfulRange(text, postComponents);

        List<Pair<Integer, Integer>> overallPosList = new ArrayList<>();
        overallPosList.addAll(nonMeaningfulRanges);

        Map<Pair<Integer, Integer>, List<String>> meaningfulDic = new HashMap<>();
        for (Triple<Integer, Integer, List<String>> each : postComponents) {
            Pair<Integer, Integer> p = Pair.of(each.getLeft(), each.getMiddle());
            overallPosList.add(p);
            meaningfulDic.put(p, each.getRight());
        }
        overallPosList.sort(Comparator.comparing(Pair::getLeft));

        List<String> returnList = new ArrayList<>();
        for (Pair<Integer, Integer> each : overallPosList) {
            if (meaningfulDic.containsKey(each)) {
                returnList.addAll(meaningfulDic.get(each));
            } else {
                Integer left = each.getLeft();
                Integer right = each.getRight();
                String substring = text.substring(left, right + 1);
                returnList.add(substring);
            }
        }
        return returnList;
    }

    private List<Set<Pair<Position, Double>>> connectedSets(List<Pair<Position, Double>> meaningfulWords) {
        DirectedGraph<Pair<Position, Double>, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);
        for (Pair<Position, Double> x : meaningfulWords) {
            graph.addVertex(x);
        }
        Set<Pair<Position, Position>> seen = new HashSet<>();
        for (Pair<Position, Double> x : meaningfulWords) {
            for (Pair<Position, Double> y : meaningfulWords) {
                if (!x.equals(y)) {
                    if (!checkIntersection(x.getLeft(), y.getLeft()).isEmpty()) {
                        Pair<Position, Position> of = Pair.of(x.getLeft(), y.getLeft());
                        if (!seen.contains(of)) {
                            graph.addEdge(x, y);
                            seen.add(of);
                        }
                    }
                }
            }
        }
        return new ConnectivityInspector<>(graph).connectedSets();
    }

    /**
     * We either calculate this each call to segment (and allow multiple lengths)
     * or at construction time, and pin to a single length. The CPU time not
     * too bad, but it's a nontrivial amount of RAM, so we do it each time.
     */
    private Pair<Map<String, Long>, Map<String, List<String>>> calculateNgrams(int minLength) {
        Map<String, Long> ngramDistribution = new HashMap<>();
        Map<String, List<String>> ngramTree = new HashMap<>();
        for (Map.Entry<String, Long> e : unigramCounts.entrySet()) {
            String entry = e.getKey();
            if (entry.length() >= minLength) {
                String cut = entry.substring(0, minLength);
                ngramDistribution.merge(cut, e.getValue(), Long::sum);

                if (!ngramTree.containsKey(cut)) {
                    ArrayList<String> v = new ArrayList<>();
                    v.add(entry);
                    ngramTree.put(cut, v);
                } else {
                    ngramTree.get(cut).add(entry);
                }
            }
        }

        return Pair.of(ngramDistribution, ngramTree);
    }

    private List<Pair<Integer, Integer>> nonMeaningfulRange(String text,
                                                            List<Triple<Integer, Integer, List<String>>> postComponents) {
        Set<Integer> meaningfulIndices = new HashSet<>();
        for (Triple<Integer, Integer, List<String>> each : postComponents) {
            meaningfulIndices.addAll(IntStream.range(each.getLeft(), each.getMiddle() + 1).boxed().collect(Collectors.toList()));
        }

        return missingSegments(text, meaningfulIndices);
    }

    ArrayList<Pair<Integer, Integer>> missingSegments(String text, Set<Integer> meaningfulIndices) {
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

        List<Pair<Pair<Position, Double>, Something>> lst = new CircularList<>();
        for (Pair<Position, Double> tu1 : meaningfulWords) {
            int pos = 0;
            ArrayList<Integer> prevList = new ArrayList<>();
            for (Pair<Position, Double> tu2 : meaningfulWords) {
                Position p1 = tu1.getLeft();
                Position p2 = tu2.getLeft();
                if (checkIntersection(p1, p2).isEmpty() && p1.start == p2.end + 1) {
                    prevList.add(pos + 1);
                }
                pos++;
            }
            if (!prevList.isEmpty()) {
                Collections.reverse(prevList);
            }
            if (prevList.isEmpty()) {
                lst.add(Pair.of(tu1, new Something(0, Optional.empty())));
            } else if (prevList.size() == 1) {
                lst.add(Pair.of(tu1, new Something(prevList.get(0), Optional.empty())));
            } else {
                lst.add(Pair.of(tu1, new Something(prevList.get(0), Optional.of(prevList.get(1)))));
            }
        }

        int j = lst.size();

        Pair<Double, HashMap<Integer, Double>> r = opt(j, lst);
        HashMap<Integer, Double> memo = r.getRight();
        Double result = r.getLeft() == null ? 0d : r.getLeft();

        List<Integer> maxV =
                memo.entrySet().stream()
                        .sorted(Comparator.comparing(Map.Entry::getValue))
                        .filter(x -> x.getValue().equals(result))
                        .map(Map.Entry::getKey).collect(Collectors.toList());
        j = maxV.get(maxV.size() - 1) + 1;

        List<Position> path = path(lst, j, memo);

        List<String> wordList = new ArrayList<>();
        for (Position position : path) {
            wordList.add(position.ngram);
        }

        int s = path.get(0).start;
        int e = lst.get(lst.size() - 1).getLeft().getLeft().end;
        return Triple.of(s, e, wordList);

    }

    private List<Position> path(List<Pair<Pair<Position, Double>, Something>> lst, int j, HashMap<Integer, Double> memo) {
        List<Position> path = new ArrayList<>();
        _path(j, path, memo, lst);
        Collections.reverse(path);
        return path;
    }

    private void _path(int j,
                       List<Position> path,
                       HashMap<Integer, Double> memo,
                       List<Pair<Pair<Position, Double>, Something>> lst) {
        if (j == 0) {
            return;
        }

        boolean b = j - 2 >= 0
                ? Objects.equals(memo.get(j - 1), memo.get(j - 2))
                : memo.get(0) != null && memo.get(0).equals(0d);
        if (b) {
            if (j != 1) {
                _path(j - 1, path, memo, lst);
            } else {
                path.add(lst.get(0).getLeft().getLeft());
            }
        } else {
            Pair<Pair<Position, Double>, Something> prev = lst.get(j - 1);
            Something thing = prev.getRight();
            int tempI = 0;
            if (!thing.end.isPresent()) {
                path.add(prev.getLeft().getLeft());
                _path(prev.getRight().get(tempI), path, memo, lst);
            } else {
                List<Double> pList = Arrays.asList(memo.get(thing.start - 1), memo.get(thing.end.get() - 1));
                Double maxP = Double.max(pList.get(0), pList.get(1));
                path.add(prev.getLeft().getLeft());
                _path(pList.get(0).equals(maxP) ? thing.start : thing.end.get(), path, memo, lst);

            }
        }
    }

    private Pair<Double, HashMap<Integer, Double>> opt(int j, List<Pair<Pair<Position, Double>, Something>> lst) {
        HashMap<Integer, Double> memo = new HashMap<>();
        Double aDouble = _opt(j, lst, memo);
        return Pair.of(aDouble, memo);
    }

    private Double _opt(int j, List<Pair<Pair<Position, Double>, Something>> lst, HashMap<Integer, Double> memo) {
        if (j == 0) return null;

        if (memo.containsKey(j - 1)) {
            return memo.get(j - 1);
        } else {
            Pair<Pair<Position, Double>, Something> next = lst.get(j - 1);
            Double max = max(
                    //choose j
                    add(_opt(next.getRight().start, lst, memo),
                            next.getLeft().getRight(),
                            penalize(j, next.getRight().start, lst)),
                    //not choose j and jump to j-1 only when nesrest overlpping word has the same finish position
                    lst.get(j - 2).getLeft().getLeft().end == next.getLeft().getLeft().end ? _opt(j - 1, lst, memo) : null
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

    private Double penalize(int current, Integer prev, List<Pair<Pair<Position, Double>, Something>> lst) {
        double penalty = -10;
        double bigramReward = 0;
        double gap;
        Pair<Pair<Position, Double>, Something> curr = lst.get(current - 1);
        Pair<Pair<Position, Double>, Something> previous = lst.get(prev - 1);

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

    public class CircularList<E> extends ArrayList<E> {
        @Override
        public E get(int i) {
            return super.get(i < 0 ? i + size() : i);
        }
    }


    class Something {
        final Integer start;
        final Optional<Integer> end;

        Something(Integer start, Optional<Integer> end) {
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
    }


    private Set<Integer> checkIntersection(Position left, Position right) {
        Set<Integer> range = IntStream.range(right.start, right.end + 1).boxed().collect(Collectors.toSet());
        return IntStream.range(left.start, left.end + 1).filter(range::contains).boxed().collect(Collectors.toSet());
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

    private List<WordPosition> cuts(int minLen, String word) {
        ArrayList<WordPosition> xs = new ArrayList<>();
        int counter = 0;
        for (int i = minLen; i < word.length() + 1; i++) {
            xs.add(new WordPosition(word.substring(counter, i), counter, counter + minLen, word.substring(i, word.length())));
            counter++;
        }
        return xs;
    }

    double getUnigramScore(String word) {
        double scale = Math.log10(1024908267229.0);
        Long x = fullUnigramCounts.get(word);
        return x != null
                ? Math.log10(x) - scale
                : Math.log10(10.0) - (scale + Math.log10(Math.pow(10, word.length())));
    }

}
