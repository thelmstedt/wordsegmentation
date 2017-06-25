package vision.trademark.nlp;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.util.Objects;

class Position<T> {
    final int start;
    final int end;
    final T ngram;

    Position(T ngram, int start, int end) {
        this.ngram = ngram;
        this.start = start;
        this.end = end;
    }


    @Override
    public String toString() {
        return new ToStringBuilder(this, ToStringStyle.NO_CLASS_NAME_STYLE)
                .append("ngram", getNgram())
                .append("start", getStart())
                .append("end", getEnd())
                .toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Position)) return false;
        Position position = (Position) o;
        return getStart() == position.getStart() &&
                getEnd() == position.getEnd() &&
                Objects.equals(getNgram(), position.getNgram());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getNgram(), getStart(), getEnd());
    }

    public int getStart() {
        return start;
    }

    public int getEnd() {
        return end;
    }

    public T getNgram() {
        return ngram;
    }

}
