package vision.trademark.nlp;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

public class WordSegmentationTest {

    private final WordSegmentation ws = new WordSegmentation();

    @Test
    public void testSegment() throws Exception {
        assertThat("all good", ws.segment(2, "universityofwashington"), is(Arrays.asList("university", "of", "washington")));
        assertThat("bad on end", ws.segment(2, "qqquniversityofwashingtonqqq"), is(Arrays.asList("qqq", "university", "of", "washington", "qqq")));
        assertThat("bad in middle", ws.segment(2, "universityqqqofwashington"), is(Arrays.asList("university", "qqq", "of", "washington")));
    }

    @Test
    public void testAShenanigans() throws Exception {
        String x = "MARGARETAREYOU";
        List<String> xs = Stream.of("margaret", "are", "you").collect(Collectors.toList());
        List<String> segment = ws.segment(2, x);

        assertThat(segment, is(xs));
    }

    //TODO still some bad splits
    @Test
    public void testLarge() throws Exception {
        String x = "MARGARETAREYOUGRIEVINGOVERGOLDENGROVEUNLEAVINGLEAVESLIKETHETHINGSOFMANYOUWITHYOURFRESHTHOUGHTSCAREFORCANYOUAHASTHEHEARTGROWSOLDERITWILLCOMETOSUCHSIGHTSCOLDERBYANDBYNORSPAREASIGHTHOUGHWORLDSOFWANWOODLEAFMEALLIEANDYETYOUWILLWEEPANDKNOWWHYNOWNOMATTERCHILDTHENAMESORROWSSPRINGSARETHESAMENORMOUTHHADNONORMINDEXPRESSEDWHATHEARTHEARDOFGHOSTGUESSEDITISTHEBLIGHTMANWASBORNFORITISMARGARETYOUMOURNFOR";
        List<String> xs = Stream.of("margaret", "are", "you", "grieving", "over", "golden", "grove", "un", "leaving", "leaves", "like", "the", "things", "of", "man", "you", "with", "your", "fresh", "thoughts", "care", "for", "can", "you", "a", "has", "the", "heart", "grows", "older", "it", "will", "come", "to", "such", "sights", "colder", "by", "and", "by", "nor", "spare", "a", "sigh", "though", "worlds", "of", "wan", "wood", "leaf", "me", "allie", "and", "yet", "you", "will", "weep", "and", "know", "why", "now", "no", "matter", "child", "the", "name", "sorrows", "springs", "are", "the", "same", "nor", "mouth", "had", "non", "or", "mind", "expressed", "what", "he", "art", "heard", "of", "ghost", "guessed", "it", "is", "the", "blight", "man", "was", "born", "for", "it", "is", "margaret", "you", "mourn", "for").collect(Collectors.toList());
        List<String> segment = ws.segment(2, x);
        System.out.println(segment);
        assertThat(String.join("\n", segment), is(String.join("\n", xs)));
    }
}
