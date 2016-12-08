package text;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class Main1
{

  @FunctionalInterface
  interface HTMLHandler
  {
    public boolean action(String path, HTML html);
  }

  public static void run(String filename, HTMLHandler handler) throws IOException {
    Zip.readFile(filename, (entry, inputStream) -> {
      String path = entry.getName();
      //System.out.println(path);
      if (entry.isDirectory()) return true;
      if (path.startsWith("__MACOSX/")) return true;
      if (path.endsWith(".DS_Store")) return true;
      if (path.endsWith(".txt")) return true;
      byte[] bytes = Zip.readBytes(inputStream);
      HTML html = new HTML(new String(bytes));
      return handler.action(path, html);
    });
  }

  public static void writeToFiles(String dir, HashMap<String, HashMap<String, String>> dictionaries) throws IOException {
    new File(dir).mkdir();

    LinkedList<String> files = new LinkedList<>();
    for (String source : dictionaries.keySet()) {
      files.addLast(source);
    }
    Collections.sort(files);

    LinkedList<String> files2 = new LinkedList<>();
    int count = 0;
    for (String source : files) {
      ++ count;
      String filename = String.format("%1$05d", count) + ".txt";
      LinkedList<String> lines = new LinkedList<>();
      for (String english : dictionaries.get(source).keySet()) {
        lines.addLast(english);
        lines.addLast(dictionaries.get(source).get(english));
      }
      files2.addLast(filename + " : " + source);
      FileHelper.writeLines(dir + "/" + filename, lines);
    }
    FileHelper.writeLines(dir + "/FILES.txt", files2);
  }

  public static void readIciba1(String filename, String outFolder) throws IOException {
    HashMap<String, HashMap<String, String>> dictionaries = new HashMap<>();

    //int[] count = new int[1];
    System.out.println("Reading file: " + filename);
    run(filename, (path, html) -> {
      //++ count[0]; if (count[0] > 500) throw new RuntimeException("500");

      ArrayList<HTML.Element> list = html.querySelectorAll("li", "class=dj_li");
      //if (list.size() > 0) System.out.println(path + " " + new Date());

      for (HTML.Element item : list) {
        HTML.Element stc_en_txt = item.querySelector("span", "class=stc_en_txt font_arial");
        HTML.Element stc_cn_txt = item.querySelector("span", "class=stc_cn_txt");
        String stc_from = item.querySelector("p", "class=stc_from fl").toString().trim();

        _removeNumber(path, stc_en_txt);
        _simplify(path, stc_en_txt);
        _simplify(path, stc_cn_txt);

        if (! dictionaries.containsKey(stc_from)) {
          dictionaries.put(stc_from, new HashMap<String, String>());
        }
        dictionaries.get(stc_from).put(_toString(stc_en_txt), _toString(stc_cn_txt));
      }
      return true;
    });

    System.out.println("Writing to folder: " + outFolder);
    writeToFiles(outFolder, dictionaries);
  }

  static Pattern numberPattern = Pattern.compile("[0-9]+\\.(\\s+)?");

  static void _removeNumber(String path, HTML.Element sentence) {
    String text = (String)sentence.children.get(0);
    text = text.trim();
    Matcher matcher = numberPattern.matcher(text);
    if (matcher.find() && matcher.start() == 0) {
      if (matcher.end() == text.length()) {
        sentence.children.remove(0);
      }
      else {
        text = text.substring(matcher.end());
        sentence.children.set(0, text);
      }
    } 
    else {
      System.err.println(path + " : number not found: " + sentence.toHTML(null));
    }
  }

  static void _simplify(String path, HTML.Element sentence) {
    Object[] children = sentence.children.toArray(new Object[0]);
    sentence.children.clear();
    for (Object c : children) {
      if (c instanceof String) {
        sentence.children.add(c);
      }
      else {
        HTML.Element e = (HTML.Element)c;
        if (e.tag.equals("span")) {
          if (e.attributes.size() != 1 || ! e.attributes.containsKey("class")) {
            //System.err.println(path + " : span format : " + e.toHTML(null) + "\r\n");
            continue;
          }

          e.tag = e.attributes.get("class");
          e.attributes.clear();
          String text = e.toString().trim();
          e.children.clear();
          e.children.add(text);
          sentence.children.add(e);
        }
        else if (e.tag.equals("em")) {
          sentence.children.add(e.toString().trim());
        }
        else {
          break;
        }
      }
    }
  }

  static String _toString(HTML.Element sentence) {
    StringBuffer buf = new StringBuffer();
    for (Object c : sentence.children) {
      buf.append(" " + c.toString().trim());
      /*
      if (c instanceof String) buf.append(" " + ((String)c).trim());
      else buf.append(" " + ((HTML.Element)c).toHTML(null));
      */
    }
    return buf.toString().trim();
  }

  public static void main(String[] args) throws IOException {
    System.out.println(new Date());
    readIciba1("data-iciba-1.zip", "iciba-1-txt");
    System.out.println(new Date());
  }

}