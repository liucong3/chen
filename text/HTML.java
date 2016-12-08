package text;

import java.io.InputStream; 
import java.io.IOException; 
import java.util.ArrayList; 
import java.util.LinkedList; 
import java.util.HashMap;
import java.util.HashSet;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class HTML {

	public static class Element
	{
		public Element parent = null;
		public String tag = null;
		public HashMap<String, String> attributes = new HashMap<>();
		public ArrayList<Object> children = new ArrayList<>(); // Element OR Text(String)
		public boolean isCompleted = false;
		public boolean isEndElement = false;

		String _beginElement() {
			StringBuffer buf = new StringBuffer();
			buf.append("<" + tag);
			for (String attr : attributes.keySet()) {
				String value = attributes.get(attr);
				value = (value.indexOf("'") == -1 ? '\'' + value + '\'' : '"' + value + '"');
				buf.append(" " + attr + "=" + value);
			}
			buf.append(children.size() == 0 ? "/>" : ">");
			return buf.toString();
		}

		public String toHTML(String tabs) {
			StringBuffer buf = new StringBuffer();
			buf.append((tabs == null ? "" : tabs) + _beginElement() + (tabs == null ? "" : "\r\n"));
			if (children.size() > 0) {
				for (Object c : children) {
					if (c instanceof String) {
						buf.append(tabs == null ? c : (tabs + ' ' + c + "\r\n"));
					}
					else {
						buf.append(((HTML.Element)c).toHTML(tabs == null ? null : tabs + ' '));
					}
				}
				buf.append(tabs == null ? "</" + tag + ">" : tabs + "</" + tag + ">\r\n");
				//if (! isCompleted) buf.append(tabs == null ? "?" : tabs + "?\r\n");
			
			}
			return buf.toString();
		}

		public String toString() {
			StringBuffer buf = new StringBuffer();
			if (children.size() > 0) {
				for (Object c : children) buf.append(c.toString());
			}
			return buf.toString();
		}

		static Pattern attributePattern = Pattern.compile("([a-zA-Z_][a-zA-Z0-9_]*)(\\s*=\\s*([^'\"\\s]+|'[^']*'|\"[^\"]*\"))?");

		public boolean parse(String text, HashSet<String> tags) {
			//System.out.println("parsing: " + text);
			text = text.substring(1, text.length() - 1);
			isEndElement = (text.charAt(0) == '/');
			if (isEndElement) {
				text = text.substring(1, text.length());
			}
			if (text.charAt(text.length() - 1) == '/') {
				isCompleted = true;
				text = text.substring(0, text.length() - 1);
			}
			//System.out.println("parsing: " + text);

			Matcher matcher = attributePattern.matcher(text);
			while (matcher.find()) {
				if (tag == null) {
					tag = matcher.group().toLowerCase();
				}
				else {
					String attr = matcher.group(1);
					String value = matcher.group(3);
					if (value == null) value = "true";
					if (value.charAt(0) == '\'' || value.charAt(0) == '"') {
						value = value.substring(1, value.length() - 1);
					}
					attributes.put(attr, value);
				}
			}

			if (tag == null || tag.contains("=")) return false;
			return (tags.size() == 0 || tags.contains(tag));
		}

		//////////////////////////////////////////

		static boolean _matchAttributes(HashMap<String, String> attributes, HashMap<String, String> toMatch) {
			for (String key : toMatch.keySet()) {
				if (! attributes.containsKey(key)) return false;
				if (! attributes.get(key).equals(toMatch.get(key))) return false;
			}
			return true;
		}

		static HashMap<String, String> _parseAttributes(String... attributeTexts) {
			HashMap<String, String> attributes = new HashMap<>();
			for (String attributeText : attributeTexts) {
				String[] pair = attributeText.split("=");
				attributes.put(pair[0].trim(), pair[1].trim());
			}
			return attributes;
		}

		public ArrayList<Element> querySelectorAll(String tag, HashMap<String, String> attributes) {
			ArrayList<Element> elements = new ArrayList<>();
			if (tag == null || this.tag.equals(tag)) {
				if (attributes == null || _matchAttributes(this.attributes, attributes)) {
					elements.add(this);
				}
			}
			for (Object node : children) {
				if (node instanceof String) continue;
				Element element = (Element)node;
				elements.addAll(element.querySelectorAll(tag, attributes));
			}
			return elements;
		}

		public ArrayList<Element> querySelectorAll(String tag, String... attributeTexts) {
			return querySelectorAll(tag, _parseAttributes(attributeTexts));
		}

		public Element querySelector(String tag, HashMap<String, String> attributes) {
			ArrayList<Element> elements = querySelectorAll(tag, attributes);
			return (elements.size() == 0 ? null : elements.get(0));
		}

		public Element querySelector(String tag, String... attributeTexts) {
			return querySelector(tag, _parseAttributes(attributeTexts));
		}

	}

	static Pattern elementPattern = Pattern.compile("<[a-zA-Z_/].*?>");

	static LinkedList<Object> _parseNodes(String webPageText, HashSet<String> tags) {
		LinkedList<Object> nodes = new LinkedList<>();
		Matcher matcher = elementPattern.matcher(webPageText);
		int pos = 0;
		int start = 0;
		while (matcher.find(start)) {
			Element element = new Element();
			String text = webPageText.substring(matcher.start(), matcher.end());
			if (element.parse(text, tags)) {
				text = webPageText.substring(pos, matcher.start());
				if (text.trim().length() > 0) nodes.add(text);
				nodes.add(element);
				start = pos = matcher.end();
			}
			else {
				++ start;
			}
		}
		String text = webPageText.substring(pos);
		if (text.trim().length() > 0) nodes.addLast(text);
		return nodes;
	}

	static String[] patternTexts = new String[] {
		"\\s+",
		"<!DOCTYPE.*?>",
		"<!--.*?-->",
		"<script(\\s|>).*?</script>",
		"<style(\\s|>).*?</style>",
		//"&[a-zA-Z]+;?",
	};

	// remove <!DOCUMENT TYPE>, <!-- ... -->, <script ... </script>, <style ... </style>, &...;
	static String _removeSpecialContent(String webPageText) {
		for (String patternText : patternTexts) {
			webPageText = webPageText.replaceAll(patternText, " ");
		}
		return webPageText;
	}

	static Element _findInRecentChildren(Element currentElement, String tag) {
		if (! currentElement.isCompleted && currentElement.tag.equals(tag)) return currentElement;
		Element found = null;
		for (Object node : currentElement.children) {
			if (node instanceof String) continue;
			Element element = (Element)node;
			Element found2 = _findInRecentChildren(element, tag);
			found = (found2 != null ? found2 : found);
		}
		return found;
	}

	static Element _findInParent(Element currentElement, String tag) {
		if (currentElement == null) return null;
		if ((! currentElement.isCompleted) && currentElement.tag.equals(tag)) return currentElement;
		//System.err.println(tag + " - " + currentElement.tag + " - " + currentElement.parent.toHTML(null));
		return _findInParent(currentElement.parent, tag);
	}

	static void _appendChildren(Element currentElement, LinkedList<Object> nodes) {
		for (Object node : nodes) {
			//System.out.println((node instanceof String ? "-> " : "=> " ) + node);
			if (node instanceof String) {
				currentElement.children.add(node);
			}
			else {
				Element elelment = (Element)node;

				if (elelment.isEndElement) {
					Element matchElement = _findInRecentChildren(currentElement, elelment.tag);
					if (matchElement != null) {
						matchElement.isCompleted = true;
					}
					else {
						matchElement = _findInParent(currentElement, elelment.tag);
						if (matchElement != null) {
							matchElement.isCompleted = true;
						}
					}
					if (matchElement != null) {
						//System.out.println(" end of element: " + matchElement.tag);
						currentElement = matchElement;
						while (currentElement.isCompleted && currentElement.parent != null)
							currentElement = currentElement.parent;
					}
				}
				else {
					currentElement.children.add(elelment);
					elelment.parent = currentElement;
					if (! elelment.isCompleted) currentElement = elelment;
				}
			}
		}
	}

	private Element documentElement = null;
	static Element _createCocumentElement() {
		Element documentElement = new Element();
		documentElement.tag = ".";
		return documentElement;
	}

	public HTML(String webPageText, String... tags) {
		webPageText = _removeSpecialContent(webPageText);

		HashSet<String> tagSet = new HashSet<>();
		for (String tag : tags) tagSet.add(tag);
		LinkedList<Object> nodes = _parseNodes(webPageText, tagSet);

		documentElement = _createCocumentElement();
		_appendChildren(documentElement, nodes);
	}

	public String toString() {
		if (documentElement == null) return "";
		return documentElement.toHTML("");
	}

	public ArrayList<Element> querySelectorAll(String tag, HashMap<String, String> attributes) {
		return documentElement.querySelectorAll(tag, attributes);
	}

	public ArrayList<Element> querySelectorAll(String tag, String... attributeTexts) {
		return documentElement.querySelectorAll(tag, attributeTexts);
	}

	public Element querySelector(String tag, HashMap<String, String> attributes) {
		return documentElement.querySelector(tag, attributes);
	}

	public Element querySelector(String tag, String... attributeTexts) {
		return documentElement.querySelector(tag, attributeTexts);
	}

	/////////////////////////////
	
	public static void main(String args[]) throws IOException {
		byte[] bytes = FileHelper.readBytes("data-iciba-1/aa/004.html");
		HTML html = new HTML(new String(bytes));
		System.out.println(html);

	}



}