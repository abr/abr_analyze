"""
Performs steps 1 and 2
setup google drive and docs access
https://developers.google.com/drive/api/v3/quickstart/python
"""
from __future__ import print_function

import os
import os.path
import pickle

from apiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class GoogleDoc:
    def __init__(
        self, title, doc_id=None, id_index_doc_path=None, print_directory=False
    ):
        # location of the text file that saves a copy of the document id,
        # title, and index. We have to manually track placement of text / images
        # through the document, so at the moment the last index is tracked so
        # the document can be appended to
        if id_index_doc_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            id_index_doc_path = dir_path + "/.google_doc_ids.txt"

        path = "/".join(id_index_doc_path.split("/")[:-2])
        id_index_doc = id_index_doc_path.split("/")[-1]

        # print('folder is: ', path)
        if not os.path.exists(path) and len(path) > 0:
            # print('folder does not exist, creating it')
            os.makedirs(path)

        # print('file is: ', id_index_doc)
        if not os.path.isfile(id_index_doc):
            # print('file does not exist, creating it')
            doc = open(id_index_doc, "w")
            doc.close()

        self.drive = None
        self.SCOPES = ["https://www.googleapis.com/auth/drive.file"]
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in.
        # TODO add note about having to confirm login and download credential file
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        "credentials.json", self.SCOPES
                    )
                except FileNotFoundError as e:
                    print(
                        "****************************************************\n"
                        + "You must allow access to your google drive first.\n"
                        + "Please go to the following link and perform steps 1 and 2\n"
                        + "https://developers.google.com/drive/api/v3/quickstart/python\n"
                        + "****************************************************"
                    )
                    raise e

                # creds = flow.run_local_server(port=0)
                creds = flow.run_console()
            # Save the credentials for the next run
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)

        self.service = build("docs", "v1", credentials=creds)

        # if no document id passed in, check out id log to see if this document
        # has been already created so we can append to it
        if doc_id is None:
            if title is None:
                title = "python_generated_file"

            with open(id_index_doc) as fp:
                line = fp.readline()
                while line:
                    txt = line.strip()
                    if len(txt) > 0:
                        txt = txt.split(":")
                        if title == txt[0]:
                            # previously created document, grab id and last index
                            doc_id = txt[1]
                            self.last_index = int(txt[2])
                            # print('loaded previous file with id: ', doc_id)
                            break

                    line = fp.readline()

            if doc_id is None:
                self.last_index = 1
                # create a new doc
                body = {"title": title}
                self.doc = self.service.documents().create(body=body).execute()
                print("Created document with title: {0}".format(self.doc.get("title")))

                doc_id = self.doc.get("documentId")
                print("New document id: {0}".format(doc_id))

                # new document, so update our id log
                with open(id_index_doc, "a") as fp:
                    fp.write("%s:%s:%s" % (title, doc_id, self.last_index))

        self.doc_id = doc_id
        self.id_index_doc = id_index_doc
        self.title = title
        self.drive = build("drive", "v3", credentials=creds)

        # Call the Drive v3 API
        results = (
            self.drive.files()
            .list(pageSize=10, fields="nextPageToken, files(id, name)")
            .execute()
        )
        items = results.get("files", [])

        if print_directory:
            if not items:
                print("No files found.")
            else:
                print("Files in directory:")
                for item in items:
                    print(u"    -{0} ({1})".format(item["name"], item["id"]))

    def _update_index_doc(self):
        txt = open(self.id_index_doc).read().splitlines()
        new_lines = []
        for line in txt:
            if line.split(":")[1] == self.doc_id:
                line = "%s:%s:%s" % (self.title, self.doc_id, self.last_index)
            new_lines.append(line)
        with open(self.id_index_doc, "w") as fp:
            new_lines = "\n".join(new_lines)
            fp.write(new_lines)

    def add_text(self, text, start_index=None, end_index=None, style_type="HEADING_1"):
        line = text

        start_index = self.last_index
        # print('saving text "%s" to index %i' % (line, start_index))

        requests = {
            "insertText": {
                "location": {
                    "index": start_index,
                },
                "text": line + "\n",
            }
        }

        self.last_index += len(line) + 1

        _ = (
            self.service.documents()
            .batchUpdate(documentId=self.doc_id, body={"requests": requests})
            .execute()
        )

        if style_type is not None:
            self.format_text(
                start_index=start_index + 1,
                end_index=self.last_index,
                style_type=style_type,
            )

        self._update_index_doc()

    def delete_file(self, service, file_id):
        """Permanently delete a file, skipping the trash.

        Args:
            service: Drive API service instance.
            file_id: ID of the file to delete.
        """
        service.files().delete(fileId=file_id).execute()

    def delete_text(self, text, start_index=None, end_index=None):
        raise NotImplementedError

    def format_text(self, start_index, end_index, style_type="HEADING_1"):
        # raise NotImplementedError
        requests = [
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": start_index, "endIndex": end_index},
                    "paragraphStyle": {
                        "namedStyleType": style_type,
                        "spaceAbove": {"magnitude": 10.0, "unit": "PT"},
                        "spaceBelow": {"magnitude": 10.0, "unit": "PT"},
                    },
                    "fields": "namedStyleType,spaceAbove,spaceBelow",
                }
            },
        ]

        _ = (
            self.service.documents()
            .batchUpdate(documentId=self.doc_id, body={"requests": requests})
            .execute()
        )

        self._update_index_doc()

    def insert_image(self, img=None, size=None):
        """
        size is in inches
        """
        if size is None:
            size = [8, 12]
        # upload our file to the drive
        file_metadata = {"name": img}
        media = MediaFileUpload(img, mimetype="image/png")
        img_file = (
            self.drive.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        img_id = img_file.get("id")

        # set permissions to share with link
        def callback(request_id, response, exception):
            if exception:
                # Handle error
                print(exception)

        batch = self.drive.new_batch_http_request(callback=callback)

        domain_permission = {
            "type": "anyone",
            "role": "reader",
        }
        batch.add(
            self.drive.permissions().create(
                fileId=img_id,
                body=domain_permission,
                fields="id",
            )
        )
        batch.execute()

        # print('image_id: ', img_id)

        # insert image into our doc
        # size is in unit PT (points) which are defined as 1/72 of an inch
        # https://developers.google.com/docs/api/reference/rest/v1/documents#Dimension
        requests = [
            {
                "insertInlineImage": {
                    "location": {"index": self.last_index},
                    "uri": "https://drive.google.com/uc?export=view&id=%s" % img_id,
                    "objectSize": {
                        "height": {"magnitude": size[0] * 72, "unit": "PT"},
                        "width": {"magnitude": size[1] * 72, "unit": "PT"},
                    },
                }
            }
        ]

        # Execute the request.
        body = {"requests": requests}
        _ = (
            self.service.documents()
            .batchUpdate(documentId=self.doc_id, body=body)
            .execute()
        )

        self.last_index += 1
        self._update_index_doc()
        # remove the file we had to temporarily upload for private access
        self.delete_file(service=self.drive, file_id=img_id)

    def create_table(self, rows, columns):
        # Insert a table at the end of the body.
        # (An empty or unspecified segmentId field indicates the document's body.)
        raise NotImplementedError
