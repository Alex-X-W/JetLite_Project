package edu.nyu.jetlite;

import edu.nyu.jetlite.tipster.*;
import java.util.List;
import java.util.Vector;

public class Sentence extends Annotation {

    public Sentence (Span s) {
	span = s;
	type = "sentence";
    }

    /**
     * Returns the token annotation starting at position <I>start</I>, or
     * <B>null</B> if no token starts at this position.
     */

    public Token tokenAt(int start, Document doc) {
        Vector annAt = doc.annotationsAt(start);
        if (annAt == null)
            return null;
        for (int i = 0; i < annAt.size(); i++) {
            Annotation ann = (Annotation) annAt.get(i);
            if (ann instanceof Token)
                return (Token) ann;
        }
        return null;
    }


}
