
package text;

import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.HashMap;
import java.util.LinkedList;

public final class Parser
{
	
	public static final class Element
	{
		public LinkedList<Element> children;
		public String text;
	}
	
	public static LinkedList<String> getLeaves(LinkedList<Element> elements) {
		LinkedList<String> list = new LinkedList<String>();
		for (Element element : elements) {
			if (element.children == null) {
				list.addLast(element.text);
			}
			else {
				list.addAll(getLeaves(element.children));
			}
		}
		return list;
	}
	
	public static final class Finder
	{
		private LinkedList<Finder> children = new LinkedList<Finder>();
		private String headPattern;
		private String tailPattern;
	}
	
	private Finder rootFinder;
	
	public Finder addFinder(Finder parent, String headPattern, String tailPattern) {
		if (headPattern == null && tailPattern == null) {
			throw new IllegalArgumentException("headPattern == null && tailPattern == null");
		}
		// set finder
		Finder finder = new Finder();
		finder.headPattern = headPattern;
		finder.tailPattern = tailPattern;
		// add to parent
		if (parent == null) {
			rootFinder = finder;
		}
		else {
			parent.children.addLast(finder);
		}
		return finder;
	}
	
	public LinkedList<Element> parse(String text) {
		return process(rootFinder, text);
	}
	
	////////////////////////
	// Implementation
	//
	
	private static HashMap<String, Pattern> patterns = new HashMap<String, Pattern>();
	
	private static Pattern getPattern(String patternText) {
		if (patterns.containsKey(patternText)) {
			return patterns.get(patternText);
		}
		Pattern pattern = Pattern.compile(patternText);
		patterns.put(patternText, pattern);
		return pattern;
	}
	
	private static int[] find(String text, String patternText, boolean exclusive) {
		if (patternText == null) {
			return null;
		}
		Pattern pattern = getPattern(patternText);
		Matcher matcher = pattern.matcher(text);
		LinkedList<Integer> list = new LinkedList<Integer>();
		while (matcher.find()) {
			int start = matcher.start();
			if (exclusive) {
				start += matcher.group().length();
			}
			list.addLast(start);
		}
		int[] indexes = new int[list.size()];
		for (int i = 0; i < indexes.length; ++ i) {
			indexes[i] = list.removeFirst();
		}
		return indexes;
	}

	private static LinkedList<Element> process(Finder finder, String text) {
		LinkedList<Element> elements = new LinkedList<Element>();
		int[] headIndexes = find(text, finder.headPattern, true);
		int[] tailIndexes = find(text, finder.tailPattern, false);
		if (tailIndexes == null) {
			for (int i = 0; i < headIndexes.length; ++ i) {
				Element element = new Element();
				int index1 = headIndexes[i];
				int index2 = (i < headIndexes.length - 1 ? headIndexes[i + 1] : text.length());
				element.text = text.substring(index1, index2);
				elements.addLast(element);
			}
		}
		else if (headIndexes == null) {
			for (int i = 0; i < tailIndexes.length; ++ i) {
				Element element = new Element();
				int index1 = (i > 0 ? tailIndexes[i - 1] : 0);
				int index2 = tailIndexes[i];
				element.text = text.substring(index1, index2);
				elements.addLast(element);
			}
		}
		else {
			int count1 = 0;
			int count2 = 0;
			while (true) {
				if (count1 == headIndexes.length) break;
				if (count2 == tailIndexes.length) break;
				while (true) {
					if (count2 == tailIndexes.length) break;
					if (headIndexes[count1] <= tailIndexes[count2]) break;
					++ count2;
				}
				if (count2 == tailIndexes.length) break;
				Element element = new Element();
				int index1 = headIndexes[count1];
				int index2 = tailIndexes[count2];
				element.text = text.substring(index1, index2);
				elements.addLast(element);
				while (true) {
					if (count1 == headIndexes.length) break;
					if (headIndexes[count1] > tailIndexes[count2]) break;
					++ count1;
				}
			}
		}
		for (Element element1 : elements) {
			for (Finder finder1 : finder.children) {
				LinkedList<Element> children = process(finder1, element1.text);
				if (element1.children == null) {
					element1.children = children;
				}
				else {
					element1.children.addAll(children);
				}
			}
		}
		return elements;
	}

	////////////////////////
	// Test
	//
	
	public static void main(String[] args) {
		String text = "sa123besa234ba456be";
		Parser parser = new Parser();
		Parser.Finder root = parser.addFinder(null, null, "e");
		parser.addFinder(root, "a", "b");
		LinkedList<Element> elements = parser.parse(text);
		LinkedList<String> lines = getLeaves(elements);
		for (String line : lines) {
			System.out.println(line);
		}
	}
	/**/
	
}