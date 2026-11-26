package android.print;

import android.os.CancellationSignal;
import android.os.ParcelFileDescriptor;
import android.print.PrintDocumentAdapter.LayoutResultCallback;
import android.print.PrintDocumentAdapter.WriteResultCallback;

/**
 * Helper located inside package android.print to access package-private
 * constructors of LayoutResultCallback and WriteResultCallback for silent PDF generation.
 */
public class PdfPrintHelper {

    public interface Callback {
        void onSuccess();
        void onFailure();
    }

    public static void printAdapterToPdf(
            final PrintDocumentAdapter adapter,
            final PrintAttributes attributes,
            final ParcelFileDescriptor pfd,
            final Callback callback) {

        adapter.onLayout(null, attributes, null, new LayoutResultCallback() {
            @Override
            public void onLayoutFinished(PrintDocumentInfo info, boolean changed) {
                adapter.onWrite(new PageRange[]{PageRange.ALL_PAGES}, pfd, new CancellationSignal(), new WriteResultCallback() {
                    @Override
                    public void onWriteFinished(PageRange[] pages) {
                        try {
                            pfd.close();
                        } catch (Exception ignored) {}
                        callback.onSuccess();
                    }

                    @Override
                    public void onWriteFailed(CharSequence error) {
                        try {
                            pfd.close();
                        } catch (Exception ignored) {}
                        callback.onFailure();
                    }
                });
            }

            @Override
            public void onLayoutFailed(CharSequence error) {
                try {
                    pfd.close();
                } catch (Exception ignored) {}
                callback.onFailure();
            }
        }, null);
    }
}
