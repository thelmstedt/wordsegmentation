package vision.trademark.nlp;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.util.Objects;
import java.util.Optional;

class MaybeRange {
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
