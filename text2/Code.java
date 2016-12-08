
package text2;

import java.io.IOException;
import java.io.InputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Arrays;
import text.Zip;
import text.FileHelper;

public class Code
{
	@FunctionalInterface
	interface Handler
	{
		public void action(String[] en, String[] ch);
	}

	static void processZipFile(Handler handler) throws IOException {
		Zip.readFile("iciba-1-txt.zip", (entry, inputStream) -> {
			String path = entry.getName();
			if (entry.isDirectory()) return true;
			if (path.startsWith("__MACOSX/")) return true;
			if (path.endsWith(".DS_Store")) return true;
			if (path.endsWith("FILES.txt")) return true;
			if (! path.endsWith(".txt")) return true;
			processFile(inputStream, handler);
			return true;
		});

	}

	static void processFile(InputStream inputStream, Handler handler) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));
		while (true) {
			String en1 = reader.readLine();
			String ch1 = reader.readLine();
			if (en1 == null || ch1 == null) break;
			en1 = removeTags(en1, false);
			ch1 = removeTags(ch1, true);
			String[] en = splitEn(en1);
			String[] ch = splitCh(ch1);
			handler.action(en, ch);
		}
	}


	static String[] splitEn(String sen) {
		String[] en = sen.split("\\s+");
		for (int i = 0; i < en.length; ++ i) {
			en[i] = en[i].toLowerCase();
		}
		return en;
		/*
		LinkedList<String> list = new LinkedList<>();
		for (String e : en) {
			e = e.trim();
			int len = e.length();
			if (len == 0) continue;
			list.addAll(splitEn1(e));
		}
		return list.toArray(new String[0]);
		*/
	}

	/*
	static String frontChars = ",.-\"?";
	static String endChars = ",.'-\"?!";
	static HashSet<Character> frontCharSet = null;
	static HashSet<Character> endCharSet = null;

	static LinkedList<String> splitEn1(String en) {
		if (frontCharSet == null) {
			frontCharSet = new HashSet<Character>();
			for (int i = 0; i < frontChars.length(); ++ i) {
				frontCharSet.add(frontChars.charAt(i));
			}
			endCharSet = new HashSet<Character>();
			for (int i = 0; i < endChars.length(); ++ i) {
				endCharSet.add(endChars.charAt(i));
			}
		}
		LinkedList<String> end = new LinkedList<>();
		int len = en.length();
		while (len > 0) {
			if (! endCharSet.contains(en.charAt(len - 1))) break;
			-- len;
			end.addFirst("" + en.charAt(len));
			en = en.substring(0, len);
		}
		LinkedList<String> front = new LinkedList<>();		
		while (len > 0) {
			if (! frontCharSet.contains(en.charAt(0))) break;
			-- len;
			front.addLast("" + en.charAt(0));
			en = en.substring(1, len + 1);
		}
		if (len > 0) front.addLast(en);
		front.addAll(end);
		return front;
	}
	*/

	static String[] splitCh(String sen) {
		String[] ch = new String[sen.length()];
		for (int i = 0; i < ch.length; ++ i) {
			ch[i] = "" + sen.charAt(i);
		}
		return ch;
	}

	static String removeTags(String text, boolean removeSpaces) {
		text = text.replaceAll("<[^>]+>", "");
		if (removeSpaces) text = text.replaceAll(" ", "");
		return text;
	}

	//////////////////////////////////////

	public static void main(String... args) throws IOException {
		System.err.println("Counting occurrences ...");
		processZipFile(Code::countWords);
		System.err.println("Assigning word IDs ...");
		writeInfo();
		System.err.println("Printing code ...");
		System.out.println(sentenceCount);
		processZipFile(Code::writeCode);
	}

	static int sentenceCount = 0;
	static HashMap<String, Integer> enCounter = new HashMap<>();
	static HashMap<String, Integer> chCounter = new HashMap<>();
	static HashMap<String, Integer> enId = new HashMap<>();
	static HashMap<String, Integer> chId = new HashMap<>();
	static HashMap<String, Integer> enLenCounter = new HashMap<>();
	static HashMap<String, Integer> chLenCounter = new HashMap<>();

	static void increaseCount(HashMap<String, Integer> counter, String word) {
		if (counter.containsKey(word)) {
			counter.put(word, counter.get(word) + 1);
		}
		else {
			counter.put(word, 1);
		}
	}

	static void countWords(String[] en, String[] ch) {
		for (int i = 0; i < en.length; ++ i) increaseCount(enCounter, en[i]);
		for (int i = 0; i < ch.length; ++ i) increaseCount(chCounter, ch[i]);
		increaseCount(enLenCounter, "" + en.length);
		increaseCount(chLenCounter, "" + ch.length);
		++ sentenceCount;
	}

	static void writeInfo() throws IOException {
		writeInfo(enCounter, "vocab.en.txt", enId);
		System.err.println("#en = " + enId.size());
		writeInfo(enLenCounter, "len.en.txt", null); // max=56
		writeInfo(chCounter, "vocab.ch.txt", chId);
		System.err.println("#ch = " + chId.size());
		writeInfo(chLenCounter, "len.ch.txt", null); // max=188
	}

	static void writeInfo(HashMap<String, Integer> counter, String filename, HashMap<String, Integer> id) throws IOException {
		String[] key = counter.keySet().toArray(new String[0]);
		Arrays.sort(key, new Comparator<String>() {
			public int compare(String t1, String t2) {
				return counter.get(t2) - counter.get(t1);
			}
		});
		if (filename != null) {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i < key.length; ++ i) {
				writer.write(key[i] + "\t" + counter.get(key[i]) + "\n");
			} 
			writer.close();
		}
		if (id != null) {
			for (int i = 0; i < key.length; ++ i) {
				id.put(key[i], i + 1);
			}
		}
	}

	static void writeCode(String[] en, String[] ch) {
		writeCode1(en, enId);
		writeCode1(ch, chId);
	}

	static void writeCode1(String[] words, HashMap<String, Integer> id) {
		System.out.print(words.length);
		for (int i = 0; i < words.length; ++ i) {
			System.out.print(" " + id.get(words[i]));
			//System.out.print(" " + words[i]);
		}
		System.out.println();
	}

}