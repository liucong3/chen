package text;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class Main3
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
			try {
				return handler.action(path, html);
			} catch (Exception ex) {
				ex.printStackTrace(System.err);
				System.out.println(path);
				System.out.println(html);
				return false;
			}
		});
	}



	public static void readIciba3(String filename, String outFolder) throws IOException {
		HashMap<String, HashMap<String, String>> changes = new HashMap<>();
		HashMap<String, HashMap<String, String[][]>> pos = new HashMap<>();
		HashMap<String, HashMap<String, String[][]>> synonym = new HashMap<>();
		HashMap<String, HashMap<String, String[][]>> antonym = new HashMap<>();
		HashMap<String, HashMap<String, String[][]>> phrases = new HashMap<>();
		HashMap<String, String> parallel = new HashMap<>();

		//int[] count = new int[1]; count[0] = 0;
		System.out.println("Reading file: " + filename);
		run(filename, (path, html) -> {
			//++ count[0]; if (count[0] > 2000) return false;
			//System.out.println(path);
			
			HTML.Element keywordElement = html.querySelector("h1", "class=keyword"); // <h1 class='keyword'>ache</h1>
			if (keywordElement == null) {
				return true;
			}
			String word = keywordElement.toString().trim();
			//if (! word.equals("access")) return true;

			HTML.Element changes1 = html.querySelector("li", "class=change");
			if (changes1 != null) {
				changes1 = changes1.querySelector("p");
				changes.put(word, _parseChanges(changes1));
			}

			// <div class='info-article'>
			ArrayList<HTML.Element> articles = html.querySelectorAll("div", "class=info-article");
			for (HTML.Element article : articles) {
				String type = article.querySelector("li", "class=current").toString(); // <li class='current'>
				if (type.equals("英英释义")) {
					pos.put(word, _parsePOS(article));
				}
				else if (type.equals("同反义词")) {
					HashMap<String, String[][]> synonym1 = new HashMap<>();
					HashMap<String, String[][]> antonym1 = new HashMap<>();
					_parseSynonymAntonym(article, synonym1, antonym1);
					if (synonym1.size() > 0) synonym.put(word, synonym1);
					if (antonym1.size() > 0) antonym.put(word, antonym1);
				}
				else if (type.equals("词组搭配")) {
					HashMap<String, String[][]> phrases1 = _parsePhrases(article, parallel);
					if (phrases1.size() > 0) phrases.put(word, phrases1);
				}
				else if (type.equals("常用俚语")) {
					//System.out.println(word + " " + type);
				}
			}

			return true;
		});

		System.out.println("Writing to folder: " + outFolder);
		_writeChanges(outFolder, changes);
		_writeParallel(outFolder, parallel);
		_write(outFolder, "pos.txt", pos);
		_write(outFolder, "synonym.txt", synonym);
		_write(outFolder, "antonym.txt", antonym);
		_write(outFolder, "phrases.txt", phrases);
	}

	//////////
	/// CHANGE

	static HashMap<String, String> _parseChanges(HTML.Element changes) {
		HashMap<String, String> map = new HashMap<>();
		String type = null;
		for (Object obj : changes.children) {
			HTML.Element elem = (HTML.Element)obj;
			if (elem.tag.equals("span")) {
				type = elem.toString().trim();
				if (! type.endsWith("：")) throw new IllegalArgumentException("Type not ends with '：'.");
				type = type.substring(0, type.length() - 1);
			}
			else {
				String to = elem.toString().trim();
				map.put(to, type);
			}
		}
		return map;
	}

	static void _writeChanges(String dir, HashMap<String, HashMap<String, String>> changes) throws IOException {
		LinkedList<String> lines = new LinkedList<>();
		for (String word : changes.keySet()) {
			for (String to : changes.get(word).keySet()) {
				String type = changes.get(word).get(to);
				lines.addLast(to + "\t" + word + "\t" + type);
			}
		}
		new File(dir).mkdir();
		FileHelper.writeLines(dir + "/changes.txt", lines);
	}

	//////////
	/// POS

	static HashMap<String, String[][]> _parsePOS(HTML.Element article) {
		HashMap<String, String[][]> map = new HashMap<>();
		ArrayList<HTML.Element> sections = article.querySelectorAll("div", "class=collins-section"); // <div class='collins-section'>
		for (HTML.Element section : sections) {
			String pos = section.querySelector("span", "class=family-english").toString(); // <span class='family-english'>Noun</span>
			ArrayList<HTML.Element> sections2 = section.querySelectorAll("div", "class=section-prep"); // <div class='section-prep'>
			ArrayList<String[]> sectionItems = new ArrayList<>();
			for (HTML.Element section2 : sections2) {
				String meaning = section2.querySelector("p", "class=family-english size-english").toString(); // <p class='family-english size-english'>
				meaning = _removeNumber(meaning);
				ArrayList<HTML.Element> items = section2.querySelectorAll("li", "class=item"); // <li class='item'>
				sectionItems.add(_toItem(meaning, items));
			}
			map.put(pos, sectionItems.toArray(new String[0][]));
		}
		return map;
	}

	static String[] _toItem(String meaning, ArrayList<HTML.Element> items) {
		String[] item = new String[items.size() + 1];
		item[0] = meaning;
		for (int i = 0; i < items.size(); ++ i) {
			item[1 + i] = items.get(i).toString().trim();
		}
		return item;
	}

	static Pattern numberPattern = Pattern.compile("[0-9]+\\.(\\s+)?");

	static String _removeNumber(String text) {
		text = text.trim();
		Matcher matcher = numberPattern.matcher(text);
		if (matcher.find() && matcher.start() == 0) {
			text = text = text.substring(matcher.end());
		}
		return text;
	}

	static void _write(String dir, String filename, HashMap<String, HashMap<String, String[][]>> allItems) throws IOException {
		LinkedList<String> lines = new LinkedList<>();
		for (String word : allItems.keySet()) {
			HashMap<String, String[][]> allItems2 = allItems.get(word);
			lines.addLast(word + "\t\t\t");
			for (String pos : allItems2.keySet()) {
				lines.addLast("\t" + pos + "\t\t");
				String[][] items = allItems2.get(pos);
				for (String[] item : items) {
					if (item[0].length() > 0) lines.addLast("\t\t" + item[0] + "\t");
					for (int i = 1; i < item.length; ++ i) {
						if (item[i].length() > 0) lines.addLast("\t\t\t" + item[i]);
					}
				}
			}
		}
		new File(dir).mkdir();
		FileHelper.writeLines(dir + "/" + filename, lines);
	}

	//////////
	/// SYNONYM & ANTONYM
	
	static void _parseSynonymAntonym(HTML.Element article, HashMap<String, String[][]> synonym, HashMap<String, String[][]> antonym) {
		// <div class='article'>
		article = article.querySelector("div", "class=article"); 
		ArrayList<HTML.Element> divs = article.querySelectorAll("div");
		boolean isSynonym = true;
		for (HTML.Element div : divs) {
			if (div.attributes.get("class").equals("opposite-word")) {
				String type = div.toString().trim();
				if (type.equals("同义词")) isSynonym = true;
				else isSynonym = false;
			}
			else if (div.attributes.get("class").equals("collins-section")) {
				// <span class='family-english'>
				HTML.Element typeElement = div.querySelector("span", "class=family-english");
				if (typeElement == null) continue;
				String type = typeElement.toString().trim();
				//<div class='section-prep'>
				ArrayList<HTML.Element> sections = div.querySelectorAll("div", "class=section-prep"); 
				ArrayList<String[]> items = new ArrayList<>();
				for (HTML.Element sectoin : sections) {
					// <p class='family-chinese size-chinese'>
					String ch_meaning = sectoin.querySelector("p", "class=family-chinese size-chinese").toString().trim(); 
					ArrayList<HTML.Element> words = sectoin.querySelectorAll("a");
					items.add(_toItem(ch_meaning, words));
				}
				if (isSynonym) synonym.put(type, items.toArray(new String[0][]));
				else antonym.put(type, items.toArray(new String[0][]));
			}
		}
	}
	
	//////////
	/// PHRASES & PARALLEL
	
	static HashMap<String, String[][]> _parsePhrases(HTML.Element article, HashMap<String, String> parallel) {
		HashMap<String, String[][]> phrases = new HashMap<>();
		//<div class='collins-section'>
		ArrayList<HTML.Element> sections = article.querySelectorAll("div", "class=collins-section");
		for (HTML.Element section : sections) {
			//<span class='family-english'>
			String phraseText = section.querySelector("span", "class=family-english").toString().trim();
			//<div class='prep-order'>
			ArrayList<HTML.Element> meanings = section.querySelectorAll("div", "class=prep-order");
			ArrayList<String[]> items = new ArrayList<>();
			for (HTML.Element meaning : meanings) {
				// <span class='family-english size-english'>
				String en_meaning_text = meaning.querySelector("span", "class=family-english size-english").toString().trim();
				// <span class='family-chinese size-chinese'>
				String ch_meaning_text = meaning.querySelector("span", "class=family-chinese size-chinese").toString().trim();
				// due to errors in the webpages from iciba.com
				en_meaning_text = en_meaning_text.substring(0, en_meaning_text.length() - ch_meaning_text.length()).trim();
				// <p class='family-english size-english'>
				ArrayList<HTML.Element> en_sentences = meaning.querySelectorAll("p", "class=family-english size-english");
				String[] en_items = _toItem(en_meaning_text, en_sentences);
				// <p class='family-chinese size-chinese'>
				ArrayList<HTML.Element> ch_sentences = meaning.querySelectorAll("p", "class=family-chinese size-chinese");
				String[] ch_items = _toItem(ch_meaning_text, ch_sentences);
				items.add(en_items);
				for (int i = 0; i < en_items.length; ++ i) {
					if (en_items[i].length() > 0 && ch_items[i].length() > 0)
						parallel.put(en_items[i], ch_items[i]);
				}
			}
			phrases.put(phraseText, items.toArray(new String[0][]));
		}
		return phrases;
	}

	static void _writeParallel(String dir, HashMap<String, String> parallel) throws IOException {
		LinkedList<String> lines = new LinkedList<>();
		for (String en : parallel.keySet()) {
			lines.addLast(en + "\t" + parallel.get(en));
		}
		new File(dir).mkdir();
		FileHelper.writeLines(dir + "/parallel.txt", lines);
	}

	//////////
	/// FIND

	public static void find(String filename, String keyword) throws IOException {
		System.out.println("Reading file: " + filename);
		run(filename, (path, html) -> {
			
			HTML.Element keywordElement = html.querySelector("h1", "class=keyword"); // <h1 class='keyword'>ache</h1>
			if (keywordElement == null) {
				return true;
			}
			String word = keywordElement.toString().trim();
			if (word.equals(keyword)) {
				System.out.println(html);
				return false;
			}

			return true;
		});

	}

	//////////
	/// PROCESS
	
	static HashSet<String> _allowed;

	static boolean _checkWord(String phrase) {
		if (_allowed == null) {
			_allowed = new HashSet<String>();
			_allowed.add("-Ones-");
			_allowed.add("-Someone-");
			_allowed.add("-Someones-");
			_allowed.add("-Something-");
			_allowed.add("-Noun-");
			_allowed.add("-Adjective-");

			_allowed.add("there's");
			_allowed.add("that's");
			_allowed.add("it's");
			_allowed.add("here's");
			_allowed.add("how's");
			_allowed.add("who's");
			_allowed.add("can't");
			_allowed.add("isn't");
			_allowed.add("don't");
			_allowed.add("won't");
			_allowed.add("shouldn't");
			_allowed.add("I'll");
			_allowed.add("I'm");
			_allowed.add("you've");
			_allowed.add("you're");

			_allowed.add("anybody's");
			_allowed.add("child's");
			_allowed.add("life's");
			_allowed.add("times'");
			_allowed.add("lion's");
			_allowed.add("pig's");
			_allowed.add("God's");
			_allowed.add("Lord's");
			_allowed.add("hell's");
			_allowed.add("harm's");
			_allowed.add("tinker's");
			_allowed.add("all's");
			_allowed.add("witch's");
			_allowed.add("more's");
			_allowed.add("pity's");
			_allowed.add("lamb's");
			_allowed.add("two's");
			_allowed.add("yesterday's");
			_allowed.add("hen's");
			_allowed.add("season's");

			_allowed.add("they're");

		}
		if (phrase.indexOf('\'') != -1) {
			if (! _allowed.contains(phrase)) return false;
		}
		if (phrase.charAt(0) == '-' && phrase.charAt(phrase.length() - 1) == '-') {
			if (! _allowed.contains(phrase)) return false;
		}
		return true;
	}

	static boolean _checkEachWord(String phrase) {
		String[] words = phrase.split("\\s+");
		for (String word : words) {
			if (! _checkWord(word)) return false;
		}
		return true;
	}

	static String _process(String phrase) {
		phrase = phrase.replaceAll("someone's", "-Someones-");
		phrase = phrase.replaceAll("someone", "-Someone-");
		phrase = phrase.replaceAll("one's", "-Ones-");
		phrase = phrase.replaceAll("something's", "-Somethings-");
		phrase = phrase.replaceAll("something", "-Something-");
		phrase = phrase.replaceAll("&I\\{.*?\\}", "");
		return phrase;
	}

	static void process(String filename) throws IOException {
		if (filename.endsWith("phrases.txt")) {
			LinkedList<String> lines = FileHelper.readLines(filename);
			for (String line : lines) {
				if (line.charAt(0) == '\t' && line.charAt(1) != '\t') {
					String phrase = line.trim();
					System.out.println(phrase);
					//harse = _process(pharse);
					//if (! _checkEachWord(pharse)) System.out.println(pharse);
				}
			}
		}
	}


	//////////
	/// MAIN

	public static void main(String[] args) throws IOException {
		if (args.length > 0) {
			if (args[0].endsWith(".txt")) {
				process(args[0]);
			}
			else {
				find("data-iciba-3.zip", args[0]);
			}
		}
		else {
			System.out.println(new Date());
			readIciba3("data-iciba-3.zip", "iciba-3-txt");
			System.out.println(new Date());
		}
	}

}