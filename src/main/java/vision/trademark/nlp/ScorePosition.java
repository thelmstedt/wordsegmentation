package vision.trademark.nlp;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.util.Objects;

class ScorePosition<T> extends Position<T> {
    private final Double score;

    ScorePosition(T ngram, Double score, int start, int end) {
        super(ngram, start, end);
        this.score = score;
    }

    public Double getScore() {
        return score;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        ScorePosition<?> that = (ScorePosition<?>) o;
        return Objects.equals(score, that.score);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), score);
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this, ToStringStyle.NO_CLASS_NAME_STYLE)
                .append("score", score)
                .append("start", start)
                .append("end", end)
                .append("ngram", ngram)
                .toString();
    }
}
