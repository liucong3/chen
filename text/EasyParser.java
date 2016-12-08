
package text;

import java.util.LinkedList;
import java.io.IOException;

public final class EasyParser
{
	private Parser parser;
	
	public EasyParser(String[][] patterns) {
		parser = new Parser();
		Parser.Finder parent = null;
		for (String[] pattern : patterns) {
			parent = parser.addFinder(parent, pattern[0], pattern[1]);
		}
	}
	
	public LinkedList<String> parse(String text) {
		LinkedList<Parser.Element> elements = parser.parse(text);
		return Parser.getLeaves(elements);
	}
	
}