package text;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.BufferedInputStream;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipEntry;
import java.util.Arrays;

public class Zip {

	@FunctionalInterface
	public interface FileHandler
	{
		public boolean action(ZipEntry entry, InputStream inputStream) throws IOException;
	}

	public static void readFile(String filename, FileHandler handler) throws IOException {
		ZipInputStream zipInputStream = new ZipInputStream(
			new BufferedInputStream(new FileInputStream(filename)));
		//int count = 0;
		while (true) {
			//++ count; if (count > 1000) break;
			ZipEntry entry = zipInputStream.getNextEntry();
			if (entry == null) break;
			if (! handler.action(entry, zipInputStream)) break;
		}
	}

	public static byte[] readBytes(InputStream inputStream) throws IOException {
		byte[] bytes = new byte[0];
		int bufferSize = 2048;
		byte[] buffer = new byte[bufferSize];
		while (true) {
			int size = inputStream.read(buffer, 0, bufferSize);
			if (size == -1) break;
			bytes = Arrays.copyOf(bytes, bytes.length + size);
			System.arraycopy(buffer, 0, bytes, bytes.length - size, size);
		}
		return bytes;
	}

	//////////////////////////////////////////////////////
	
	public static void upZip(String filename) throws IOException {
		Zip.readFile(filename, (entry, inputStream) -> {
			String path = entry.getName();
			if (path.startsWith("__MACOSX/")) return true;
			if (path.endsWith(".DS_Store")) return true;
			boolean isDirectory = entry.isDirectory();
			createDirectory(path, isDirectory);
			if (! isDirectory) {
				byte[] bytes = readBytes(inputStream);
				writeBytes(path, bytes);
			}
			return true;
		});
	}

	public static void createDirectory(String path, boolean isDirectory) {
		int index = path.lastIndexOf("/");
		if (index != -1) createDirectory(path.substring(0, index), true);
		if (isDirectory) new File(path).mkdir();
	}

	public static void writeBytes(String filename, byte[] bytes) throws IOException {
		FileOutputStream output = new FileOutputStream(filename);
		output.write(bytes);
		output.close();
	}

	//////////////////////////////////////////////////////

	public static void main(String args[]) throws IOException {
		upZip(args[0]);
	}
}