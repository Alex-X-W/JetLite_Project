// Load from pretrained python Tensorflow event tagging model
// Author: Xuan Wang

package edu.nyu.jetlite;

import com.oracle.javafx.jmx.json.JSONException;
import edu.nyu.jetlite.tipster.*;
import java.io.*;
import java.util.*;

import org.apache.commons.lang.ArrayUtils;
import org.json.simple.*;
import org.json.simple.parser.JSONParser;
import java.util.stream.Collectors;
import java.util.Map.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 *  Identify ACE events.
 *  <p>
 *  This 'skeleton' tagger does not include finding event arguments.
 */

public class NeuralTagModel {
    private byte[] graphDef;
    private JSONObject mappings;
    private Map<Integer, String> idToChar, idToTag, idToWord;
    private Map<String, Integer> charToId, TagToId, wordToId;
    private int sentenceLength;

    public NeuralTagModel(Properties config) throws IOException {
        String modelDir = config.getProperty("NeuralTagModel.modelDir");
        String modelFileName = config.getProperty("NeuralTagModel.modelFileName");
        String modelMappings = config.getProperty("NeuralTagModel.mappings");

        // read model mappings
        JSONParser parser = new JSONParser();
        try {
            Object obj = parser.parse(new FileReader(modelMappings));
            JSONObject jsonObject = (JSONObject) obj;

            idToChar = readJsonMap(jsonObject, "id_to_char");
            idToTag = readJsonMap(jsonObject, "id_to_tag");
            idToWord = readJsonMap(jsonObject, "id_to_word");

            charToId = invert(idToChar);
            TagToId = invert(idToTag);
            wordToId = invert(idToWord);

        } catch (Exception e) {
            System.out.println("Failed reading TF model config mappings...");
            e.printStackTrace();
        }

        graphDef = readAllBytesOrExit(Paths.get(modelDir, modelFileName));
    }

    public static void main (String[] args) throws IOException {
        String[] testSentence = new String[]{"This", "morning", ",", "an", "American", "bomb", "destroyed", "a", "convoy",
                "carrying", "high", "officials"};
        Properties p = new Properties();
        p.setProperty("NeuralTagModel.modelDir", "./src/main/resources/tf_model");
        p.setProperty("NeuralTagModel.modelFileName", "optimized_NeuralTagModel.pb");
        p.setProperty("NeuralTagModel.mappings", "./src/main/resources/tf_model/mappings.json");

        NeuralTagModel neuralTagModel = new NeuralTagModel(p);
        String[] tags = neuralTagModel.predict(testSentence);
        for (int i = 0; i < tags.length; ++i) {
            System.out.println(tags[i]);
        }
    }

    private byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    public String[] predict(Document doc, Sentence sent) {
        Tensor[] input = createInput(doc, sent);
        int[][] scores = executeGraph(graphDef, input);
        String[] tags = convertToTags(scores);
        return tags;
    }

    // for testing
    public String[] predict(String[] sent) {
        Tensor[] input = createInput(sent);
        int[][] scores = executeGraph(graphDef, input);
        String[] tags = convertToTags(scores);
        return tags;
    }

    private int[][] executeGraph(byte[] graphDef, Tensor[] input) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                 Tensor result = s.runner().feed("word_ids", input[0]).feed("word_pos_ids", input[1]).
                         feed("char_for_ids", input[2]).feed("char_rev_ids", input[3]).feed("char_pos_ids", input[4]).
                         fetch("output/TensorArrayStack/TensorArrayGatherV3").run().get(0)) {
                return result.copyTo(new int[1][sentenceLength + 2]);
            }
        }
    }

    private String[] convertToTags(int[][] scores) {
        // crf decoding
        int[] score = scores[0];
        /*
        word_pos = input_[1][x] + 2
        y_pred = f_score[1:word_pos]
        words = batch_words[x]
        y_preds = [model.id_to_tag[pred] for pred in y_pred]
         */
        int wordPos = wordPosIds[0] + 2;
        int[] yPred = new int[wordPos - 1];
        for (int i = 1; i < wordPos; ++i) {
            yPred[i - 1] = score[i];
        }
        String[] ret = new String[wordPos - 1];
        for (int i = 0; i < wordPos - 1; ++i) {
            ret[i] = idToTag.get(yPred[i]);
        }
        return ret;
    }

    /*
    feed_dict_[model.word_ids]     = input_[0]
    feed_dict_[model.word_pos_ids] = input_[1]
    feed_dict_[model.char_for_ids] = input_[2]
    feed_dict_[model.char_rev_ids] = input_[3]
    feed_dict_[model.char_pos_ids] = input_[4]
    self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids') # shape:[batch_size, max_word_len]
    self.word_pos_ids = tf.placeholder(tf.int32, shape=[None], name='word_pos_ids') # shape: [batch_size]
    self.char_for_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_for_ids') # shape: [batch_size, word_max_len, char_max_len]
    self.char_rev_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_rev_ids') # shape: [batch_size, word_max_len, char_max_len]
    self.char_pos_ids = tf.placeholder(tf.int32, shape=[None, None], name='char_pos_ids') # shape: [batch_size*word_max_len, char_max_len]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    */
    private ArrayList<String> tokens;
    private ArrayList<Integer> tokenIds;
    private int[][] wordIds;
    private int[] wordPosIds;
    private int[][][] charForIds;
    private int[][][] charRevIds;
    private int[][] charPosIds;

    private Tensor[] createInput(Document doc, Sentence sent) {
        Span span = sent.span();
        int posn = span.start();

        // retrieve tokens from sentence span
        tokens = new ArrayList<>();
        tokenIds = new ArrayList<>();
        int charMaxLen = 0;
        while (posn < span.end()) {
            Annotation tokenAnnotation = doc.tokenAt(posn);
            if (tokenAnnotation == null)
                break;
            String tokenText = doc.normalizedText(tokenAnnotation).trim();
            tokens.add(tokenText);
            if (tokenText.length() > charMaxLen) {
                charMaxLen = tokenText.length();
            }
            Integer tokenId = wordToId.get(tokenText);
            tokenIds.add(tokenId == null ? wordToId.get("<UNK>"): tokenId);
            posn = tokenAnnotation.end();
        }
        sentenceLength = tokenIds.size();
        wordIds = new int[1][tokenIds.size()];
        wordIds[0] = ArrayUtils.toPrimitive(tokenIds.toArray(new Integer[0]));
        wordPosIds = new int[1];
        wordPosIds[0] = tokenIds.size() - 1;

        charForIds = new int[1][tokenIds.size()][charMaxLen];
        charRevIds = new int[1][tokenIds.size()][charMaxLen];
        charPosIds = new int[tokenIds.size()][2];

        int cnt = 0;
        for (String e: tokens) {
            int pos = e.length() - 1;
            charPosIds[cnt][0] = cnt;
            charPosIds[cnt][1] = pos;

            for (int i = 0; i < e.length(); ++i) {
                Integer charId = charToId.get(Character.toString(e.charAt(i)));
                charForIds[0][cnt][i] = charId == null ? 0: charId;
                charRevIds[0][cnt][e.length() - 1 - i] = charId == null ? 0: charId;
            }
        }

        // ids -> Tensors
        Tensor inputWordIds = makeTensor(wordIds, "word_ids");
        Tensor inputWordPosIds = makeTensor(wordPosIds, "word_pos_ids");
        Tensor inputCharForIds = makeTensor(charForIds, "char_for_ids");
        Tensor inputCharRevIds = makeTensor(charRevIds, "char_rev_ids");
        Tensor inputCharPosIds = makeTensor(charPosIds, "char_pos_ids");

        return new Tensor[]{inputWordIds, inputWordPosIds, inputCharForIds, inputCharRevIds, inputCharPosIds};
    }

    // for testing
    private Tensor[] createInput(String[] sent) {

        // retrieve tokens from sentence span
        tokens = new ArrayList<>();
        tokenIds = new ArrayList<>();
        int charMaxLen = 0;
        for (int i = 0; i < sent.length; ++i) {
            String tokenText = sent[i];
            tokens.add(tokenText);
            if (tokenText.length() > charMaxLen) {
                charMaxLen = tokenText.length();
            }
            Integer tokenId = wordToId.get(tokenText);
            tokenIds.add(tokenId == null ? wordToId.get("<UNK>"): tokenId);
        }
        sentenceLength = tokenIds.size();
        wordIds = new int[1][tokenIds.size()];
        wordIds[0] = ArrayUtils.toPrimitive(tokenIds.toArray(new Integer[0]));
        wordPosIds = new int[1];
        wordPosIds[0] = tokenIds.size() - 1;

        charForIds = new int[1][tokenIds.size()][charMaxLen];
        charRevIds = new int[1][tokenIds.size()][charMaxLen];
        charPosIds = new int[tokenIds.size()][2];

        int cnt = 0;
        for (String e: tokens) {
            int pos = e.length() - 1;
            charPosIds[cnt][0] = cnt;
            charPosIds[cnt][1] = pos;

            for (int i = 0; i < e.length(); ++i) {
                Integer charId = charToId.get(Character.toString(e.charAt(i)));
                charForIds[0][cnt][i] = charId == null ? 0: charId;
                charRevIds[0][cnt][e.length() - 1 - i] = charId == null ? 0: charId;
            }
        }

        // ids -> Tensors
        Tensor inputWordIds = makeTensor(wordIds, "word_ids");
        Tensor inputWordPosIds = makeTensor(wordPosIds, "word_pos_ids");
        Tensor inputCharForIds = makeTensor(charForIds, "char_for_ids");
        Tensor inputCharRevIds = makeTensor(charRevIds, "char_rev_ids");
        Tensor inputCharPosIds = makeTensor(charPosIds, "char_pos_ids");

        return new Tensor[]{inputWordIds, inputWordPosIds, inputCharForIds, inputCharRevIds, inputCharPosIds};
    }



    private <T> Tensor makeTensor(T input, String name) {
        try (Graph g = new Graph()) {
            tf_intergration.GraphBuilder b = new tf_intergration.GraphBuilder(g);
            final Output constWordIds = b.constant(name, input);
            try (Session s = new Session(g)) {
                return s.runner().fetch(constWordIds.op().name()).run().get(0);
            }
        }
    }

    private static <V, K> Map<V, K> invert(Map<K, V> map) {

        Map<V, K> inv = new HashMap<V, K>();

        for (Entry<K, V> entry : map.entrySet())
            inv.put(entry.getValue(), entry.getKey());

        return inv;
    }

    private void printMap(Map mp) {
        Iterator it = mp.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
            System.out.println(pair.getKey() + " = " + pair.getValue());
            it.remove(); // avoids a ConcurrentModificationException
        }
    }

    private void printType(Object obj) {
        System.out.println(obj.getClass().getName());
    }

    private Map<Integer, String> readJsonMap(JSONObject jsonObject, String fn) {
        Map<String, String> tmp = (Map<String, String>) jsonObject.get(fn);
        Map<Integer, String> ret = new HashMap<>();
        Iterator it = tmp.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, String> pair = (Map.Entry)it.next();
            ret.put(Integer.parseInt(pair.getKey()), pair.getValue());
            it.remove(); // avoids a ConcurrentModificationException
        }
        return ret;
    }
}
