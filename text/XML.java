package text;

import java.io.InputStream; 
import java.io.IOException; 
import java.util.ArrayList; 
import java.util.HashMap;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory; 
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document; 
import org.w3c.dom.Element; 
import org.w3c.dom.Node;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList; 
import org.xml.sax.SAXException;

public class XML {

	private Document _document;

	public XML(InputStream inputStream) throws IOException {
		try {
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			//factory.setValidating(true);
			DocumentBuilder builder = factory.newDocumentBuilder();
			_document = builder.parse(inputStream);
		} catch (ParserConfigurationException ex) {
			throw new IOException(ex.getMessage());
		} catch (SAXException ex) {
			throw new IOException(ex.getMessage());
		}
	}

	public Document getDocument() { return _document; }

	public ArrayList<Element> querySelectorAll(String tag, HashMap<String, String> attributes) {
		return querySelectorAll(_document.getDocumentElement(), tag, attributes);
	}

	public ArrayList<Element> querySelectorAll(String tag, String... attributeTexts) {
		return querySelectorAll(_document.getDocumentElement(), tag, attributeTexts);
	}

	public Element querySelector(String tag, HashMap<String, String> attributes) {
		return querySelector(_document.getDocumentElement(), tag, attributes);
	}
	
	public Element querySelector(String tag, String... attributeTexts) {
		return querySelector(_document.getDocumentElement(), tag, attributeTexts);
	}

	///////////////////////////////////////////////////////////////////

	public static HashMap<String, String> getAttributes(Element element) {
		HashMap<String, String> attributes = new HashMap<>();
		NamedNodeMap namedNodeMap = element.getAttributes();
		for (int i = 0; i < namedNodeMap.getLength(); ++ i) {
			Node node = namedNodeMap.item(i);
			attributes.put(node.getNodeName(), node.getNodeValue());
		}
		return attributes;
	}

	public static boolean matchAttributes(HashMap<String, String> attributes, HashMap<String, String> toMatch) {
		for (String key : toMatch.keySet()) {
			if (! attributes.containsKey(key)) return false;
			if (! attributes.get(key).equals(toMatch.get(key))) return false;
		}
		return true;
	}

	public static ArrayList<Element> querySelectorAll(Element element, String tag, HashMap<String, String> attributes) {
		ArrayList<Element> elements = new ArrayList<>();
		if (tag == null || element.getTagName().equals(tag)) {
			if (attributes == null || matchAttributes(getAttributes(element), attributes)) {
				elements.add(element);
			}
		}
		NodeList nodeList = element.getChildNodes();
		for (int i = 0; i < nodeList.getLength(); i++) {
			Node node = nodeList.item(i);
			if (node.getNodeType() != Node.ELEMENT_NODE) continue;
			elements.addAll(querySelectorAll((Element)node, tag, attributes));
		}
		return elements;
	}

	public static ArrayList<Element> querySelectorAll(Element element, String tag, String... attributeTexts) {
		HashMap<String, String> attributes = new HashMap<>();
		for (String attributeText : attributeTexts) {
			String[] pair = attributeText.split("=");
			attributes.put(pair[0].trim(), pair[1].trim());
		}
		return querySelectorAll(element, tag, attributes);
	}

	public static Element querySelector(Element element, String tag, HashMap<String, String> attributes) {
		ArrayList<Element> elements = querySelectorAll(element, tag, attributes);
		if (elements.size() == 0) return null;
		return elements.get(0);
	}

	public static Element querySelector(Element element, String tag, String... attributeTexts) {
		ArrayList<Element> elements = querySelectorAll(element, tag, attributeTexts);
		if (elements.size() == 0) return null;
		return elements.get(0);
	}

	///////////////////////////////////////////////////////////////////

	public static void main(String[] args) throws IOException {
		byte[] bytes = FileHelper.readBytes("text/apple.html");
		InputStream inputStream = new java.io.ByteArrayInputStream(bytes);
		XML xml = new XML(inputStream);
		Element element = xml.querySelector("p", "align=center");
		System.out.println(element.getTextContent());
	}

}