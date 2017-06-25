package vision.trademark.nlp;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.util.Objects;

class SuffixedPosition extends Position<String> {
    private final String suffix;

    SuffixedPosition(String ngram, int start, int end, String suffix) {
        super(ngram, start, end);
        this.suffix = suffix;
    }

    public String getSuffix() {
        return this.suffix;
    }

    public String getPrefix() {
        return super.getNgram();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        SuffixedPosition that = (SuffixedPosition) o;
        return Objects.equals(suffix, that.suffix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), suffix);
    }

    @Override
    public String toString() {
        return new ToStringBuilder(this, ToStringStyle.NO_CLASS_NAME_STYLE)
                .append("suffix", suffix)
                .append("start", start)
                .append("end", end)
                .append("ngram", ngram)
                .toString();
    }
}
