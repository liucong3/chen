package text;

import java.io.File;
import java.io.IOException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.LinkedList;

public final class FileHelper
{
	public static String ENCODING = "UTF-8";

	public static LinkedList<String> readLines(String filename) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), ENCODING));
		LinkedList<String> lines = new LinkedList<String>();
		while (true) {
			String line = reader.readLine();
			if (line == null) break;
			if (line.length() == 0) continue;
			lines.addLast(line);
		}
		reader.close();
		return lines;
	}

	public static void writeLines(String filename, LinkedList<String> lines) throws IOException {
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), ENCODING));
		for (String line : lines) {
			writer.write(line + "\n");
		}
		writer.close();
	}
	
	public static byte[] readBytes(String filename) throws IOException {
		File file = new File(filename);
		FileInputStream input = new FileInputStream(file);
		int length = (int)file.length();
		byte[] bytes = new byte[length];
		input.read(bytes);
		input.close();
		return bytes;
	}

	public static void writeBytes(String filename, byte[] bytes) throws IOException {
		FileOutputStream output = new FileOutputStream(filename);
		output.write(bytes);
		output.close();
	}

}